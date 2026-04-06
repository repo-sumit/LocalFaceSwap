#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <gdiplus.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr auto kUploadsPollInterval = std::chrono::milliseconds(750);

struct AppConfig {
    int cameraIndex = 0;
    int width = 640;
    int height = 480;
    int fps = 30;
    int detectInterval = 6;
    bool mirror = true;
    bool debug = false;
    std::string overlay = "face";
    std::string backend = "dshow";
    std::string uploadsDir = "uploads";
};

struct FaceTrackState {
    bool hasFace = false;
    cv::Rect2f box;
    cv::Rect2f smoothedBox;
    std::vector<cv::Point2f> points;
    cv::Mat previousGray;
    int framesSinceRefresh = 0;
};

struct AvatarAsset {
    bool loaded = false;
    std::filesystem::path path;
    std::filesystem::file_time_type modifiedAt{};
    cv::Mat bgr;
    cv::Mat alpha;
    cv::Rect foregroundBox;
    cv::Point2f faceAnchor;
    float estimatedHeadHeight = 1.0f;
    cv::Rect faceSwapBox;
    cv::Mat faceSwapBgr;
    cv::Mat faceSwapAlpha;
};

struct AvatarTuning {
    float scaleMultiplier = 1.0f;
    float offsetX = 0.0f;
    float offsetY = 0.0f;
};

struct UploadSelection {
    std::filesystem::path path;
    std::filesystem::file_time_type modifiedAt{};
};

enum class UploadReloadState {
    Unchanged,
    Loaded,
    Missing,
    Failed,
};

struct UploadReloadResult {
    UploadReloadState state = UploadReloadState::Unchanged;
    std::filesystem::path path;
};

struct LoadedRaster {
    cv::Mat raw;
    bool hasAlpha = false;
};

std::optional<LoadedRaster> loadRasterImage(const std::filesystem::path& filePath);

class FrameGrabber {
public:
    bool open(const AppConfig& config) {
        int backend = cv::CAP_ANY;
        if (config.backend == "dshow" && cv::CAP_DSHOW != 0) {
            backend = cv::CAP_DSHOW;
        }

        if (backend == cv::CAP_ANY) {
            capture_.open(config.cameraIndex);
        } else {
            capture_.open(config.cameraIndex, backend);
        }

        if (!capture_.isOpened()) {
            return false;
        }

        capture_.set(cv::CAP_PROP_FRAME_WIDTH, config.width);
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, config.height);
        capture_.set(cv::CAP_PROP_FPS, config.fps);
        if (cv::CAP_PROP_BUFFERSIZE != 0) {
            capture_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }
        return true;
    }

    void start() {
        running_ = true;
        worker_ = std::thread([this]() { run(); });
    }

    void stop() {
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        capture_.release();
    }

    bool latestFrame(cv::Mat& outFrame, std::uint64_t& outFrameId) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latestFrame_.empty()) {
            return false;
        }
        outFrame = latestFrame_.clone();
        outFrameId = frameId_;
        return true;
    }

private:
    void run() {
        cv::Mat frame;
        while (running_) {
            if (!capture_.read(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            std::lock_guard<std::mutex> lock(mutex_);
            latestFrame_ = frame.clone();
            ++frameId_;
        }
    }

    cv::VideoCapture capture_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    std::mutex mutex_;
    cv::Mat latestFrame_;
    std::uint64_t frameId_ = 0;
};

cv::Rect clampRect(const cv::Rect2f& rect, const cv::Size& bounds) {
    const float x1 = std::clamp(rect.x, 0.0f, static_cast<float>(bounds.width - 1));
    const float y1 = std::clamp(rect.y, 0.0f, static_cast<float>(bounds.height - 1));
    const float x2 = std::clamp(rect.x + rect.width, x1 + 1.0f, static_cast<float>(bounds.width));
    const float y2 = std::clamp(rect.y + rect.height, y1 + 1.0f, static_cast<float>(bounds.height));
    return cv::Rect(cv::Point2f(x1, y1), cv::Point2f(x2, y2));
}

float lerp(float a, float b, float alpha) {
    return a + (b - a) * alpha;
}

cv::Rect2f smoothRect(const cv::Rect2f& current, const cv::Rect2f& target, float alpha) {
    return {
        lerp(current.x, target.x, alpha),
        lerp(current.y, target.y, alpha),
        lerp(current.width, target.width, alpha),
        lerp(current.height, target.height, alpha),
    };
}

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

std::vector<cv::Point2f> seedTrackingPoints(const cv::Mat& gray, const cv::Rect& box) {
    const cv::Rect clipped = clampRect(box, gray.size());
    if (clipped.width < 10 || clipped.height < 10) {
        return {};
    }

    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(0));
    cv::rectangle(mask, clipped, cv::Scalar(255), cv::FILLED);

    std::vector<cv::Point2f> points;
    cv::goodFeaturesToTrack(gray, points, 80, 0.01, 6.0, mask, 3, false, 0.04);
    return points;
}

cv::Rect2f makeGuideBox(const cv::Size& frameSize) {
    const float width = static_cast<float>(frameSize.width) * 0.34f;
    const float height = static_cast<float>(frameSize.height) * 0.48f;
    const float x = (static_cast<float>(frameSize.width) - width) * 0.5f;
    const float y = (static_cast<float>(frameSize.height) - height) * 0.22f;
    return {x, y, width, height};
}

bool lockFaceFromGuide(const cv::Mat& gray, FaceTrackState& state, const cv::Rect2f& guideBox) {
    state.box = clampRect(guideBox, gray.size());
    state.smoothedBox = state.box;
    state.points = seedTrackingPoints(gray, state.box);
    state.previousGray = gray.clone();
    state.framesSinceRefresh = 0;
    state.hasFace = state.points.size() >= 8;
    return state.hasFace;
}

bool updateTrackedFace(const cv::Mat& previousGray, const cv::Mat& gray, FaceTrackState& state) {
    if (state.points.size() < 8 || previousGray.empty()) {
        return false;
    }

    std::vector<cv::Point2f> nextPoints;
    std::vector<unsigned char> status;
    std::vector<float> error;
    cv::calcOpticalFlowPyrLK(
        previousGray,
        gray,
        state.points,
        nextPoints,
        status,
        error,
        cv::Size(21, 21),
        3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03)
    );

    std::vector<cv::Point2f> oldGood;
    std::vector<cv::Point2f> newGood;
    oldGood.reserve(state.points.size());
    newGood.reserve(state.points.size());
    for (std::size_t i = 0; i < status.size(); ++i) {
        if (status[i]) {
            oldGood.push_back(state.points[i]);
            newGood.push_back(nextPoints[i]);
        }
    }

    if (newGood.size() < 8) {
        return false;
    }

    std::vector<double> dxValues;
    std::vector<double> dyValues;
    dxValues.reserve(newGood.size());
    dyValues.reserve(newGood.size());
    for (std::size_t i = 0; i < newGood.size(); ++i) {
        dxValues.push_back(static_cast<double>(newGood[i].x - oldGood[i].x));
        dyValues.push_back(static_cast<double>(newGood[i].y - oldGood[i].y));
    }

    const double dx = median(dxValues);
    const double dy = median(dyValues);

    const cv::Rect2f oldBounds = cv::boundingRect(oldGood);
    const cv::Rect2f newBounds = cv::boundingRect(newGood);
    const float scaleX = oldBounds.width > 1.0f ? newBounds.width / oldBounds.width : 1.0f;
    const float scaleY = oldBounds.height > 1.0f ? newBounds.height / oldBounds.height : 1.0f;
    const float scale = std::clamp((scaleX + scaleY) * 0.5f, 0.85f, 1.15f);

    cv::Rect2f updated = state.box;
    updated.x = static_cast<float>(updated.x + dx);
    updated.y = static_cast<float>(updated.y + dy);
    updated.width *= scale;
    updated.height *= scale;

    const cv::Point2f center(updated.x + updated.width * 0.5f, updated.y + updated.height * 0.5f);
    updated.width = std::clamp(updated.width, 40.0f, static_cast<float>(gray.cols));
    updated.height = std::clamp(updated.height, 40.0f, static_cast<float>(gray.rows));
    updated.x = center.x - updated.width * 0.5f;
    updated.y = center.y - updated.height * 0.5f;

    state.box = clampRect(updated, gray.size());
    state.points = newGood;
    return state.points.size() >= 8;
}

void alphaBlend(const cv::Mat& overlay, const cv::Mat& mask, cv::Mat& target) {
    cv::Mat overlayFloat;
    cv::Mat targetFloat;
    cv::Mat alphaFloat;
    overlay.convertTo(overlayFloat, CV_32FC3);
    target.convertTo(targetFloat, CV_32FC3);
    mask.convertTo(alphaFloat, CV_32FC1, 1.0 / 255.0);

    cv::Mat alphaChannels[3] = {alphaFloat, alphaFloat, alphaFloat};
    cv::Mat alphaMerged;
    cv::merge(alphaChannels, 3, alphaMerged);

    cv::Mat blended = overlayFloat.mul(alphaMerged) + targetFloat.mul(cv::Scalar::all(1.0) - alphaMerged);
    blended.convertTo(target, CV_8UC3);
}

std::string lowerCopy(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool isSupportedImage(const std::filesystem::path& path) {
    const std::string ext = lowerCopy(path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp" || ext == ".webp";
}

bool directoryHasSupportedImages(const std::filesystem::path& dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        return false;
    }

    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && isSupportedImage(entry.path())) {
            return true;
        }
    }
    return false;
}

std::optional<std::filesystem::path> findProjectRoot(const std::filesystem::path& start) {
    std::filesystem::path current = start;
    for (int depth = 0; depth < 6; ++depth) {
        if (std::filesystem::exists(current / "vcpkg.json") && std::filesystem::exists(current / "native")) {
            return current;
        }
        if (!current.has_parent_path()) {
            break;
        }
        current = current.parent_path();
    }
    return std::nullopt;
}

std::filesystem::path resolveUploadsDir(const std::filesystem::path& configuredDir, const std::filesystem::path& executablePath) {
    if (configuredDir.is_absolute()) {
        std::filesystem::create_directories(configuredDir);
        return configuredDir;
    }

    const std::filesystem::path cwdCandidate = std::filesystem::current_path() / configuredDir;
    const std::filesystem::path exeDir = executablePath.parent_path();
    const std::optional<std::filesystem::path> projectRoot = findProjectRoot(exeDir);

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(cwdCandidate);
    if (projectRoot.has_value()) {
        candidates.push_back(*projectRoot / configuredDir);
    }
    candidates.push_back(exeDir / configuredDir);

    std::vector<std::filesystem::path> uniqueCandidates;
    for (const auto& candidate : candidates) {
        const std::filesystem::path absoluteCandidate = std::filesystem::weakly_canonical(candidate.parent_path()) / candidate.filename();
        if (std::find(uniqueCandidates.begin(), uniqueCandidates.end(), absoluteCandidate) == uniqueCandidates.end()) {
            uniqueCandidates.push_back(absoluteCandidate);
        }
    }

    for (const auto& candidate : uniqueCandidates) {
        if (directoryHasSupportedImages(candidate)) {
            std::filesystem::create_directories(candidate);
            return candidate;
        }
    }

    if (projectRoot.has_value()) {
        const std::filesystem::path preferred = *projectRoot / configuredDir;
        std::filesystem::create_directories(preferred);
        return preferred;
    }

    std::filesystem::create_directories(cwdCandidate);
    return cwdCandidate;
}

cv::Mat estimateForegroundMask(const cv::Mat& bgr) {
    const int sample = std::max(4, std::min(bgr.cols, bgr.rows) / 20);
    const auto patchMean = [&](int x, int y) {
        const cv::Rect roi(x, y, std::min(sample, bgr.cols - x), std::min(sample, bgr.rows - y));
        return cv::mean(bgr(roi));
    };

    const cv::Scalar c1 = patchMean(0, 0);
    const cv::Scalar c2 = patchMean(std::max(0, bgr.cols - sample), 0);
    const cv::Scalar c3 = patchMean(0, std::max(0, bgr.rows - sample));
    const cv::Scalar c4 = patchMean(std::max(0, bgr.cols - sample), std::max(0, bgr.rows - sample));
    const cv::Scalar background(
        (c1[0] + c2[0] + c3[0] + c4[0]) * 0.25,
        (c1[1] + c2[1] + c3[1] + c4[1]) * 0.25,
        (c1[2] + c2[2] + c3[2] + c4[2]) * 0.25
    );

    cv::Mat diff;
    cv::absdiff(bgr, background, diff);
    cv::Mat grayDiff;
    cv::cvtColor(diff, grayDiff, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(grayDiff, mask, 28, 255, cv::THRESH_BINARY);

    const int kernelSize = std::max(3, ((std::min(bgr.cols, bgr.rows) / 80) | 1));
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernelSize, kernelSize));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 1);
    cv::GaussianBlur(mask, mask, cv::Size(0, 0), 2.2);
    return mask;
}

cv::Rect findForegroundBox(const cv::Mat& alpha) {
    cv::Mat thresholded;
    cv::threshold(alpha, thresholded, 16, 255, cv::THRESH_BINARY);
    std::vector<cv::Point> points;
    cv::findNonZero(thresholded, points);
    if (points.empty()) {
        return {0, 0, alpha.cols, alpha.rows};
    }
    return cv::boundingRect(points);
}

cv::Rect estimateFaceSwapBox(const cv::Rect& foregroundBox, const cv::Size& bounds) {
    if (foregroundBox.width < 2 || foregroundBox.height < 2) {
        return {0, 0, bounds.width, bounds.height};
    }

    const float portraitAspect = static_cast<float>(foregroundBox.height) / std::max(1, foregroundBox.width);
    cv::Rect2f estimated;

    if (portraitAspect >= 1.35f) {
        const float width = foregroundBox.width * 0.58f;
        const float height = std::max(foregroundBox.height * 0.28f, std::min(foregroundBox.height * 0.42f, width * 1.28f));
        estimated = {
            foregroundBox.x + (foregroundBox.width - width) * 0.5f,
            foregroundBox.y + foregroundBox.height * 0.03f,
            width,
            height,
        };
    } else {
        estimated = {
            foregroundBox.x + foregroundBox.width * 0.06f,
            foregroundBox.y + foregroundBox.height * 0.05f,
            foregroundBox.width * 0.88f,
            foregroundBox.height * 0.88f,
        };
    }

    return clampRect(estimated, bounds);
}

cv::Mat makeSoftFaceMask(const cv::Size& size) {
    if (size.width < 2 || size.height < 2) {
        return {};
    }

    cv::Mat mask(size, CV_8UC1, cv::Scalar(0));
    cv::ellipse(
        mask,
        cv::Point(size.width / 2, size.height / 2),
        cv::Size(
            std::max(1, static_cast<int>(std::round(size.width * 0.46f))),
            std::max(1, static_cast<int>(std::round(size.height * 0.49f)))
        ),
        0.0,
        0.0,
        360.0,
        cv::Scalar(255),
        cv::FILLED,
        cv::LINE_AA
    );

    const int blurKernel = std::max(3, ((std::min(size.width, size.height) / 6) | 1));
    cv::GaussianBlur(mask, mask, cv::Size(blurKernel, blurKernel), 0.0);
    return mask;
}

void populateFaceSwapData(AvatarAsset& asset) {
    asset.faceSwapBox = estimateFaceSwapBox(asset.foregroundBox, asset.alpha.size());
    if (asset.faceSwapBox.width < 4 || asset.faceSwapBox.height < 4) {
        asset.faceSwapBox = cv::Rect(0, 0, asset.alpha.cols, asset.alpha.rows);
    }

    asset.faceSwapBgr = asset.bgr(asset.faceSwapBox).clone();
    asset.faceSwapAlpha = asset.alpha(asset.faceSwapBox).clone();

    cv::Mat softMask = makeSoftFaceMask(asset.faceSwapBox.size());
    if (!softMask.empty()) {
        cv::multiply(asset.faceSwapAlpha, softMask, asset.faceSwapAlpha, 1.0 / 255.0);
    }
    cv::GaussianBlur(asset.faceSwapAlpha, asset.faceSwapAlpha, cv::Size(0, 0), 1.4);
}

std::optional<AvatarAsset> loadAvatarFromFile(
    const std::filesystem::path& filePath,
    const std::filesystem::file_time_type& modifiedAt
) {
    const std::optional<LoadedRaster> raster = loadRasterImage(filePath);
    if (!raster.has_value() || raster->raw.empty()) {
        return std::nullopt;
    }

    const cv::Mat& raw = raster->raw;

    AvatarAsset asset;
    asset.loaded = true;
    asset.path = filePath;
    asset.modifiedAt = modifiedAt;

    if (raster->hasAlpha && raw.channels() == 4) {
        std::vector<cv::Mat> channels;
        cv::split(raw, channels);
        asset.alpha = channels[3];
        cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, asset.bgr);
    } else if (raw.channels() == 4) {
        cv::cvtColor(raw, asset.bgr, cv::COLOR_BGRA2BGR);
        asset.alpha = estimateForegroundMask(asset.bgr);
    } else if (raw.channels() == 3) {
        asset.bgr = raw;
        asset.alpha = estimateForegroundMask(asset.bgr);
    } else if (raw.channels() == 1) {
        cv::cvtColor(raw, asset.bgr, cv::COLOR_GRAY2BGR);
        asset.alpha = cv::Mat(raw.size(), CV_8UC1, cv::Scalar(255));
    } else {
        return std::nullopt;
    }

    asset.foregroundBox = findForegroundBox(asset.alpha);
    asset.estimatedHeadHeight = std::max(1.0f, static_cast<float>(asset.foregroundBox.height) * 0.26f);
    asset.faceAnchor = {
        asset.foregroundBox.x + asset.foregroundBox.width * 0.5f,
        asset.foregroundBox.y + asset.estimatedHeadHeight * 0.58f
    };

    populateFaceSwapData(asset);
    return asset;
}

std::optional<UploadSelection> findLatestUploadImage(const std::filesystem::path& uploadsDir) {
    std::filesystem::create_directories(uploadsDir);

    std::optional<UploadSelection> newest;
    for (const auto& entry : std::filesystem::directory_iterator(uploadsDir)) {
        if (!entry.is_regular_file() || !isSupportedImage(entry.path())) {
            continue;
        }

        const UploadSelection candidate{entry.path(), entry.last_write_time()};
        if (!newest.has_value() ||
            candidate.modifiedAt > newest->modifiedAt ||
            (candidate.modifiedAt == newest->modifiedAt &&
             candidate.path.filename().string() > newest->path.filename().string())) {
            newest = candidate;
        }
    }

    return newest;
}

bool sameUploadSelection(const std::optional<UploadSelection>& left, const std::optional<UploadSelection>& right) {
    if (left.has_value() != right.has_value()) {
        return false;
    }
    if (!left.has_value()) {
        return true;
    }
    return left->path == right->path && left->modifiedAt == right->modifiedAt;
}

UploadReloadResult reloadAvatarFromUploads(
    const std::filesystem::path& uploadsDir,
    bool force,
    std::optional<UploadSelection>& activeSelection,
    std::optional<AvatarAsset>& avatar
) {
    const std::optional<UploadSelection> nextSelection = findLatestUploadImage(uploadsDir);
    if (!force && sameUploadSelection(activeSelection, nextSelection)) {
        return {};
    }

    activeSelection = nextSelection;
    if (!nextSelection.has_value()) {
        avatar.reset();
        return {UploadReloadState::Missing, {}};
    }

    const std::optional<AvatarAsset> loaded = loadAvatarFromFile(nextSelection->path, nextSelection->modifiedAt);
    if (!loaded.has_value()) {
        avatar.reset();
        return {UploadReloadState::Failed, nextSelection->path};
    }

    avatar = loaded;
    return {UploadReloadState::Loaded, nextSelection->path};
}

#if defined(_WIN32)
class GdiPlusSession {
public:
    static bool ready() {
        static GdiPlusSession session;
        return session.ready_;
    }

private:
    GdiPlusSession() {
        Gdiplus::GdiplusStartupInput startupInput;
        ready_ = Gdiplus::GdiplusStartup(&token_, &startupInput, nullptr) == Gdiplus::Ok;
    }

    ~GdiPlusSession() {
        if (ready_) {
            Gdiplus::GdiplusShutdown(token_);
        }
    }

    ULONG_PTR token_ = 0;
    bool ready_ = false;
};

std::optional<LoadedRaster> loadImageWithGdiPlus(const std::filesystem::path& filePath) {
    if (!GdiPlusSession::ready()) {
        return std::nullopt;
    }

    Gdiplus::Bitmap source(filePath.wstring().c_str());
    if (source.GetLastStatus() != Gdiplus::Ok) {
        return std::nullopt;
    }

    const int width = static_cast<int>(source.GetWidth());
    const int height = static_cast<int>(source.GetHeight());
    if (width < 1 || height < 1) {
        return std::nullopt;
    }

    Gdiplus::Bitmap converted(width, height, PixelFormat32bppARGB);
    if (converted.GetLastStatus() != Gdiplus::Ok) {
        return std::nullopt;
    }

    {
        Gdiplus::Graphics graphics(&converted);
        if (graphics.GetLastStatus() != Gdiplus::Ok) {
            return std::nullopt;
        }
        graphics.DrawImage(&source, 0, 0, width, height);
        if (graphics.GetLastStatus() != Gdiplus::Ok) {
            return std::nullopt;
        }
    }

    Gdiplus::Rect rect(0, 0, width, height);
    Gdiplus::BitmapData bitmapData{};
    if (converted.LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat32bppARGB, &bitmapData) != Gdiplus::Ok) {
        return std::nullopt;
    }

    cv::Mat bgra(height, width, CV_8UC4);
    const auto* sourceBytes = static_cast<const std::uint8_t*>(bitmapData.Scan0);
    const int rowStride = std::abs(bitmapData.Stride);
    for (int row = 0; row < height; ++row) {
        const int sourceRow = bitmapData.Stride >= 0 ? row : (height - 1 - row);
        std::memcpy(bgra.ptr(row), sourceBytes + (sourceRow * rowStride), static_cast<std::size_t>(width) * 4);
    }

    converted.UnlockBits(&bitmapData);

    LoadedRaster raster;
    raster.raw = bgra;
    raster.hasAlpha = (source.GetFlags() & Gdiplus::ImageFlagsHasAlpha) != 0;
    return raster;
}
#endif

std::optional<LoadedRaster> loadRasterImage(const std::filesystem::path& filePath) {
    cv::Mat raw = cv::imread(filePath.string(), cv::IMREAD_UNCHANGED);
    if (!raw.empty()) {
        return LoadedRaster{raw, raw.channels() == 4};
    }

#if defined(_WIN32)
    return loadImageWithGdiPlus(filePath);
#else
    return std::nullopt;
#endif
}

void blendImageAt(cv::Mat& frame, const cv::Mat& overlayBgr, const cv::Mat& alpha, const cv::Point2f& topLeft) {
    const int x = static_cast<int>(std::round(topLeft.x));
    const int y = static_cast<int>(std::round(topLeft.y));

    const int left = std::max(0, x);
    const int top = std::max(0, y);
    const int right = std::min(frame.cols, x + overlayBgr.cols);
    const int bottom = std::min(frame.rows, y + overlayBgr.rows);
    if (left >= right || top >= bottom) {
        return;
    }

    const cv::Rect frameRoi(left, top, right - left, bottom - top);
    const cv::Rect overlayRoi(left - x, top - y, frameRoi.width, frameRoi.height);

    cv::Mat target = frame(frameRoi);
    alphaBlend(overlayBgr(overlayRoi), alpha(overlayRoi), target);
}

cv::Mat matchOverlayToTarget(const cv::Mat& overlayBgr, const cv::Mat& mask, const cv::Mat& targetBgr) {
    if (overlayBgr.empty()) {
        return {};
    }

    const cv::Scalar overlayMean = cv::mean(overlayBgr, mask);
    const cv::Scalar targetMean = cv::mean(targetBgr);

    std::vector<cv::Mat> channels;
    cv::split(overlayBgr, channels);
    for (int channel = 0; channel < 3; ++channel) {
        const double sourceValue = std::max(overlayMean[channel], 1.0);
        const double gain = std::clamp(targetMean[channel] / sourceValue, 0.72, 1.38);
        const double shift = std::clamp((targetMean[channel] - overlayMean[channel]) * 0.18, -16.0, 16.0);
        channels[channel].convertTo(channels[channel], CV_8UC1, gain, shift);
    }

    cv::Mat adjusted;
    cv::merge(channels, adjusted);
    return adjusted;
}

void blendColorMatchedImageAt(cv::Mat& frame, const cv::Mat& overlayBgr, const cv::Mat& alpha, const cv::Point2f& topLeft) {
    const int x = static_cast<int>(std::round(topLeft.x));
    const int y = static_cast<int>(std::round(topLeft.y));

    const int left = std::max(0, x);
    const int top = std::max(0, y);
    const int right = std::min(frame.cols, x + overlayBgr.cols);
    const int bottom = std::min(frame.rows, y + overlayBgr.rows);
    if (left >= right || top >= bottom) {
        return;
    }

    const cv::Rect frameRoi(left, top, right - left, bottom - top);
    const cv::Rect overlayRoi(left - x, top - y, frameRoi.width, frameRoi.height);

    cv::Mat target = frame(frameRoi);
    const cv::Mat overlaySection = overlayBgr(overlayRoi);
    const cv::Mat alphaSection = alpha(overlayRoi);
    const cv::Mat matched = matchOverlayToTarget(overlaySection, alphaSection, target);
    alphaBlend(matched, alphaSection, target);
}

void drawHatOverlay(cv::Mat& frame, const cv::Rect2f& face) {
    const float hatWidth = face.width * 1.55f;
    const float hatHeight = face.height * 0.80f;
    const float hatX = face.x + face.width * 0.5f - hatWidth * 0.5f;
    const float hatY = face.y - hatHeight * 0.85f;
    const cv::Rect region = clampRect(cv::Rect2f(hatX, hatY, hatWidth, hatHeight), frame.size());
    if (region.width < 8 || region.height < 8) {
        return;
    }

    cv::Mat roi = frame(region);
    cv::Mat overlay = roi.clone();
    cv::Mat mask(roi.size(), CV_8UC1, cv::Scalar(0));

    const auto toLocal = [&](float px, float py) {
        return cv::Point(static_cast<int>(std::round(px - region.x)), static_cast<int>(std::round(py - region.y)));
    };

    const cv::Point brimCenter = toLocal(face.x + face.width * 0.5f, face.y - face.height * 0.02f);
    const cv::Size brimAxes(
        static_cast<int>(std::round(hatWidth * 0.42f)),
        static_cast<int>(std::round(hatHeight * 0.10f))
    );
    const cv::Point crownCenter = toLocal(face.x + face.width * 0.5f, face.y - hatHeight * 0.35f);
    const cv::Size crownAxes(
        static_cast<int>(std::round(hatWidth * 0.26f)),
        static_cast<int>(std::round(hatHeight * 0.26f))
    );

    cv::ellipse(overlay, crownCenter, crownAxes, 0.0, 0.0, 360.0, cv::Scalar(30, 30, 30), cv::FILLED, cv::LINE_AA);
    cv::ellipse(mask, crownCenter, crownAxes, 0.0, 0.0, 360.0, cv::Scalar(215), cv::FILLED, cv::LINE_AA);

    cv::ellipse(overlay, brimCenter, brimAxes, 0.0, 0.0, 360.0, cv::Scalar(18, 18, 18), cv::FILLED, cv::LINE_AA);
    cv::ellipse(mask, brimCenter, brimAxes, 0.0, 0.0, 360.0, cv::Scalar(230), cv::FILLED, cv::LINE_AA);

    alphaBlend(overlay, mask, roi);
}

void drawGlassesOverlay(cv::Mat& frame, const cv::Rect2f& face) {
    const float glassesWidth = face.width * 0.94f;
    const float glassesHeight = face.height * 0.24f;
    const float glassesX = face.x + face.width * 0.5f - glassesWidth * 0.5f;
    const float glassesY = face.y + face.height * 0.34f;
    const cv::Rect region = clampRect(cv::Rect2f(glassesX, glassesY, glassesWidth, glassesHeight), frame.size());
    if (region.width < 8 || region.height < 8) {
        return;
    }

    cv::Mat roi = frame(region);
    cv::Mat overlay = roi.clone();
    cv::Mat mask(roi.size(), CV_8UC1, cv::Scalar(0));

    const int width = region.width;
    const int height = region.height;
    const cv::Rect leftLens(static_cast<int>(width * 0.06f), static_cast<int>(height * 0.16f), static_cast<int>(width * 0.36f), static_cast<int>(height * 0.68f));
    const cv::Rect rightLens(static_cast<int>(width * 0.58f), static_cast<int>(height * 0.16f), static_cast<int>(width * 0.36f), static_cast<int>(height * 0.68f));
    const cv::Rect bridge(static_cast<int>(width * 0.44f), static_cast<int>(height * 0.38f), static_cast<int>(width * 0.12f), static_cast<int>(height * 0.16f));

    cv::rectangle(overlay, leftLens, cv::Scalar(42, 22, 12), cv::FILLED, cv::LINE_AA);
    cv::rectangle(overlay, rightLens, cv::Scalar(42, 22, 12), cv::FILLED, cv::LINE_AA);
    cv::rectangle(overlay, bridge, cv::Scalar(36, 18, 10), cv::FILLED, cv::LINE_AA);

    cv::rectangle(mask, leftLens, cv::Scalar(165), cv::FILLED, cv::LINE_AA);
    cv::rectangle(mask, rightLens, cv::Scalar(165), cv::FILLED, cv::LINE_AA);
    cv::rectangle(mask, bridge, cv::Scalar(180), cv::FILLED, cv::LINE_AA);

    cv::rectangle(overlay, leftLens, cv::Scalar(95, 160, 210), 3, cv::LINE_AA);
    cv::rectangle(overlay, rightLens, cv::Scalar(95, 160, 210), 3, cv::LINE_AA);
    cv::rectangle(mask, leftLens, cv::Scalar(235), 2, cv::LINE_AA);
    cv::rectangle(mask, rightLens, cv::Scalar(235), 2, cv::LINE_AA);

    alphaBlend(overlay, mask, roi);
}

void drawFaceSwapOverlay(cv::Mat& frame, const cv::Rect2f& face, const AvatarAsset& asset, const AvatarTuning& tuning) {
    if (!asset.loaded || asset.faceSwapBgr.empty() || asset.faceSwapAlpha.empty()) {
        return;
    }

    const float sourceAspect = static_cast<float>(asset.faceSwapBgr.cols) / std::max(1, asset.faceSwapBgr.rows);
    const float containerWidth = std::max(40.0f, face.width * 0.86f * tuning.scaleMultiplier);
    const float containerHeight = std::max(52.0f, face.height * 0.96f * tuning.scaleMultiplier);

    float targetWidth = containerWidth;
    float targetHeight = targetWidth / std::max(0.1f, sourceAspect);
    if (targetHeight < containerHeight) {
        targetHeight = containerHeight;
        targetWidth = targetHeight * sourceAspect;
    }

    cv::Mat scaledBgr;
    cv::Mat scaledAlpha;
    const cv::Size targetSize(
        std::max(1, static_cast<int>(std::round(targetWidth))),
        std::max(1, static_cast<int>(std::round(targetHeight)))
    );
    cv::resize(asset.faceSwapBgr, scaledBgr, targetSize, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(asset.faceSwapAlpha, scaledAlpha, targetSize, 0.0, 0.0, cv::INTER_LINEAR);

    const cv::Point2f center(
        face.x + face.width * 0.5f + face.width * tuning.offsetX,
        face.y + face.height * 0.52f + face.height * tuning.offsetY
    );
    const cv::Point2f topLeft(center.x - targetWidth * 0.5f, center.y - targetHeight * 0.5f);

    blendColorMatchedImageAt(frame, scaledBgr, scaledAlpha, topLeft);
}

void drawAvatarOverlay(cv::Mat& frame, const cv::Rect2f& face, const AvatarAsset& asset, const AvatarTuning& tuning) {
    if (!asset.loaded || asset.bgr.empty() || asset.alpha.empty()) {
        return;
    }

    const float liveHeadHeight = std::max(1.0f, face.height * 1.55f);
    const float scale = std::max(0.05f, (liveHeadHeight / asset.estimatedHeadHeight) * tuning.scaleMultiplier);

    cv::Mat scaledBgr;
    cv::Mat scaledAlpha;
    cv::resize(asset.bgr, scaledBgr, cv::Size(), scale, scale, cv::INTER_LINEAR);
    cv::resize(asset.alpha, scaledAlpha, cv::Size(), scale, scale, cv::INTER_LINEAR);

    const cv::Point2f liveFaceCenter(face.x + face.width * 0.5f, face.y + face.height * 0.5f);
    const cv::Point2f anchor(asset.faceAnchor.x * scale, asset.faceAnchor.y * scale);
    const cv::Point2f topLeft(
        liveFaceCenter.x - anchor.x + face.width * tuning.offsetX,
        liveFaceCenter.y - anchor.y + face.height * tuning.offsetY
    );

    blendImageAt(frame, scaledBgr, scaledAlpha, topLeft);
}

void drawDebugOverlay(cv::Mat& frame, const cv::Rect2f& face) {
    cv::rectangle(frame, clampRect(face, frame.size()), cv::Scalar(40, 220, 120), 2, cv::LINE_AA);
}

void applyOverlay(cv::Mat& frame, const AppConfig& config, const cv::Rect2f& face, const AvatarAsset& asset, const AvatarTuning& tuning) {
    if (config.overlay == "hat") {
        drawHatOverlay(frame, face);
    } else if (config.overlay == "glasses") {
        drawGlassesOverlay(frame, face);
    } else if (config.overlay == "avatar") {
        drawAvatarOverlay(frame, face, asset, tuning);
    } else if (config.overlay == "face") {
        drawFaceSwapOverlay(frame, face, asset, tuning);
    }

    if (config.debug) {
        drawDebugOverlay(frame, face);
    }
}

void drawOutlinedText(cv::Mat& frame, const std::string& text, const cv::Point& origin, double scale) {
    cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(20, 20, 20), 3, cv::LINE_AA);
    cv::putText(frame, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(240, 240, 240), 1, cv::LINE_AA);
}

AppConfig parseArgs(int argc, char** argv) {
    const cv::String keys =
        "{help h usage ?||Show help}"
        "{camera|0|Camera index}"
        "{width|640|Capture width}"
        "{height|480|Capture height}"
        "{fps|30|Requested camera FPS}"
        "{detect-interval|6|Tracked frames before point refresh}"
        "{overlay|face|Overlay type: face, avatar, hat, or glasses}"
        "{backend|dshow|Camera backend: dshow or any}"
        "{debug|false|Show debug rectangle}"
        "{mirror|true|Mirror the selfie preview}"
        "{uploads-dir|uploads|Folder containing the uploaded avatar image}";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("local_face_filter");
    if (parser.has("help")) {
        parser.printMessage();
        std::exit(0);
    }

    AppConfig config;
    config.cameraIndex = parser.get<int>("camera");
    config.width = parser.get<int>("width");
    config.height = parser.get<int>("height");
    config.fps = parser.get<int>("fps");
    config.detectInterval = std::max(1, parser.get<int>("detect-interval"));
    config.overlay = parser.get<std::string>("overlay");
    config.backend = parser.get<std::string>("backend");
    config.debug = parser.get<bool>("debug");
    config.mirror = parser.get<bool>("mirror");
    config.uploadsDir = parser.get<std::string>("uploads-dir");

    if (!parser.check()) {
        parser.printErrors();
        std::exit(1);
    }

    if (config.overlay != "face" && config.overlay != "avatar" && config.overlay != "hat" && config.overlay != "glasses") {
        std::cerr << "overlay must be 'face', 'avatar', 'hat', or 'glasses'\n";
        std::exit(1);
    }

    if (config.backend != "dshow" && config.backend != "any") {
        std::cerr << "backend must be 'dshow' or 'any'\n";
        std::exit(1);
    }

    return config;
}

void printUploadMessage(const UploadReloadResult& result, const std::filesystem::path& uploadsDir, bool manual) {
    if (result.state == UploadReloadState::Loaded) {
        std::cout << (manual ? "Loaded image: " : "Auto-loaded image: ") << result.path << '\n';
    } else if (result.state == UploadReloadState::Missing) {
        if (manual) {
            std::cout << "No JPG/PNG found in " << uploadsDir << ". Drop one into the folder and try again.\n";
        } else {
            std::cout << "Uploads folder is empty. Drop a JPG/PNG into " << uploadsDir << ".\n";
        }
    } else if (result.state == UploadReloadState::Failed) {
        std::cout << "Found image but failed to load: " << result.path << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    cv::setUseOptimized(true);

    const AppConfig config = parseArgs(argc, argv);
    const std::filesystem::path executablePath = std::filesystem::absolute(std::filesystem::path(argv[0]));
    const std::filesystem::path uploadsDir = resolveUploadsDir(std::filesystem::path(config.uploadsDir), executablePath);

    std::optional<AvatarAsset> avatar;
    std::optional<UploadSelection> activeUploadSelection;
    const UploadReloadResult initialLoad = reloadAvatarFromUploads(uploadsDir, true, activeUploadSelection, avatar);
    AvatarTuning avatarTuning;

    FrameGrabber grabber;
    if (!grabber.open(config)) {
        std::cerr << "Failed to open camera.\n";
        return 1;
    }

    grabber.start();

    FaceTrackState trackState;
    std::uint64_t lastProcessedId = 0;
    double fpsEstimate = 0.0;
    auto previousFrameTime = Clock::now();
    auto nextUploadsPoll = Clock::now() + kUploadsPollInterval;

    std::cout << "LocalFaceSwap Native running.\n";
    std::cout << "Watching uploads folder: " << uploadsDir << '\n';
    if (initialLoad.state == UploadReloadState::Loaded) {
        std::cout << "Starting with image: " << initialLoad.path << '\n';
    } else {
        std::cout << "No JPG/PNG found yet. Drop one into the uploads folder and it will auto-load.\n";
    }
    std::cout << "Keys: q quit | space lock | c clear | r rescan now | 1 hat | 2 glasses | 3 face swap | 4 avatar | [ ] scale | i k vertical | j l horizontal | d debug\n";

    AppConfig runtimeConfig = config;
    const AvatarAsset emptyAsset;

    while (true) {
        if (Clock::now() >= nextUploadsPoll) {
            nextUploadsPoll = Clock::now() + kUploadsPollInterval;
            const UploadReloadResult autoReload = reloadAvatarFromUploads(uploadsDir, false, activeUploadSelection, avatar);
            printUploadMessage(autoReload, uploadsDir, false);
        }

        cv::Mat frame;
        std::uint64_t frameId = 0;
        if (!grabber.latestFrame(frame, frameId) || frameId == lastProcessedId) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        lastProcessedId = frameId;

        if (runtimeConfig.mirror) {
            cv::flip(frame, frame, 1);
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);
        const cv::Rect2f guideBox = makeGuideBox(frame.size());

        bool tracked = false;
        if (trackState.hasFace) {
            tracked = updateTrackedFace(trackState.previousGray, gray, trackState);
        }

        if (tracked) {
            trackState.smoothedBox = smoothRect(trackState.smoothedBox, trackState.box, 0.22f);
            ++trackState.framesSinceRefresh;
            if (trackState.framesSinceRefresh >= runtimeConfig.detectInterval) {
                trackState.points = seedTrackingPoints(gray, trackState.box);
                trackState.framesSinceRefresh = 0;
                if (trackState.points.size() < 8) {
                    trackState.hasFace = false;
                    trackState.points.clear();
                }
            }
        } else if (trackState.hasFace) {
            trackState.hasFace = false;
            trackState.points.clear();
        }

        trackState.previousGray = gray.clone();

        if (trackState.hasFace) {
            const AvatarAsset& activeAsset = avatar.has_value() ? *avatar : emptyAsset;
            applyOverlay(frame, runtimeConfig, trackState.smoothedBox, activeAsset, avatarTuning);
        } else {
            const cv::Rect guideRect = clampRect(guideBox, frame.size());
            cv::rectangle(frame, guideRect, cv::Scalar(60, 220, 255), 2, cv::LINE_AA);
            drawOutlinedText(frame, "Center face and press SPACE", cv::Point(guideRect.x, std::max(30, guideRect.y - 12)), 0.65);
        }

        if ((runtimeConfig.overlay == "face" || runtimeConfig.overlay == "avatar") && !avatar.has_value()) {
            drawOutlinedText(frame, "Drop a JPG/PNG into uploads/ to start", cv::Point(16, frame.rows - 24), 0.55);
        }

        const auto now = Clock::now();
        const double seconds = std::chrono::duration<double>(now - previousFrameTime).count();
        previousFrameTime = now;
        if (seconds > 0.0) {
            const double instantFps = 1.0 / seconds;
            fpsEstimate = fpsEstimate <= 0.0 ? instantFps : (fpsEstimate * 0.90) + (instantFps * 0.10);
        }

        drawOutlinedText(frame, "native-swap", cv::Point(16, 28), 0.7);

        std::string stats = "fps " + std::to_string(static_cast<int>(std::round(fpsEstimate))) +
            "  overlay " + runtimeConfig.overlay;
        if (avatar.has_value()) {
            stats += "  img " + avatar->path.filename().string();
        }
        drawOutlinedText(frame, stats, cv::Point(16, 54), 0.55);

        if (runtimeConfig.overlay == "face" || runtimeConfig.overlay == "avatar") {
            const std::string label = runtimeConfig.overlay == "face" ? "face" : "avatar";
            const std::string tune =
                label + " scale " + std::to_string(static_cast<int>(std::round(avatarTuning.scaleMultiplier * 100.0f))) +
                "%  offset(" + std::to_string(static_cast<int>(std::round(avatarTuning.offsetX * 100.0f))) +
                "," + std::to_string(static_cast<int>(std::round(avatarTuning.offsetY * 100.0f))) + ")";
            drawOutlinedText(frame, tune, cv::Point(16, 80), 0.5);
        }

        cv::imshow("LocalFaceSwap Native", frame);

        const int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) {
            break;
        }
        if (key == ' ') {
            if (!lockFaceFromGuide(gray, trackState, guideBox)) {
                std::cout << "Face lock failed. Move closer or improve lighting, then try again.\n";
            }
        }
        if (key == 'c') {
            trackState = FaceTrackState{};
        }
        if (key == 'r') {
            const UploadReloadResult reload = reloadAvatarFromUploads(uploadsDir, true, activeUploadSelection, avatar);
            printUploadMessage(reload, uploadsDir, true);
        }
        if (key == '1') {
            runtimeConfig.overlay = "hat";
        }
        if (key == '2') {
            runtimeConfig.overlay = "glasses";
        }
        if (key == '3') {
            runtimeConfig.overlay = "face";
        }
        if (key == '4') {
            runtimeConfig.overlay = "avatar";
        }
        if (key == 'd') {
            runtimeConfig.debug = !runtimeConfig.debug;
        }
        if (key == '[') {
            avatarTuning.scaleMultiplier = std::max(0.4f, avatarTuning.scaleMultiplier - 0.05f);
        }
        if (key == ']') {
            avatarTuning.scaleMultiplier = std::min(2.2f, avatarTuning.scaleMultiplier + 0.05f);
        }
        if (key == 'i') {
            avatarTuning.offsetY -= 0.03f;
        }
        if (key == 'k') {
            avatarTuning.offsetY += 0.03f;
        }
        if (key == 'j') {
            avatarTuning.offsetX -= 0.03f;
        }
        if (key == 'l') {
            avatarTuning.offsetX += 0.03f;
        }
    }

    grabber.stop();
    cv::destroyAllWindows();
    return 0;
}
