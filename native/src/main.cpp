#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct AppConfig {
    int cameraIndex = 0;
    int width = 640;
    int height = 480;
    int fps = 30;
    int detectInterval = 4;
    bool mirror = true;
    bool debug = false;
    std::string overlay = "hat";
    std::string backend = "dshow";
};

struct FaceTrackState {
    bool hasFace = false;
    cv::Rect2f box;
    cv::Rect2f smoothedBox;
    std::vector<cv::Point2f> points;
    cv::Mat previousGray;
    int framesSinceDetection = 1000;
};

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
    state.framesSinceDetection = 0;
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
    state.points = seedTrackingPoints(gray, state.box);
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

    const auto toLocal = [&](float x, float y) {
        return cv::Point(static_cast<int>(std::round(x - region.x)), static_cast<int>(std::round(y - region.y)));
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

    cv::ellipse(
        overlay,
        cv::Point(crownCenter.x, crownCenter.y - static_cast<int>(std::round(hatHeight * 0.04f))),
        cv::Size(static_cast<int>(std::round(crownAxes.width * 0.82f)), static_cast<int>(std::round(crownAxes.height * 0.48f))),
        0.0,
        200.0,
        340.0,
        cv::Scalar(70, 70, 70),
        2,
        cv::LINE_AA
    );

    cv::ellipse(
        overlay,
        cv::Point(crownCenter.x, crownCenter.y + static_cast<int>(std::round(crownAxes.height * 0.32f))),
        cv::Size(static_cast<int>(std::round(crownAxes.width * 0.92f)), static_cast<int>(std::round(crownAxes.height * 0.16f))),
        0.0,
        0.0,
        360.0,
        cv::Scalar(35, 35, 110),
        cv::FILLED,
        cv::LINE_AA
    );
    cv::ellipse(
        mask,
        cv::Point(crownCenter.x, crownCenter.y + static_cast<int>(std::round(crownAxes.height * 0.32f))),
        cv::Size(static_cast<int>(std::round(crownAxes.width * 0.92f)), static_cast<int>(std::round(crownAxes.height * 0.16f))),
        0.0,
        0.0,
        360.0,
        cv::Scalar(200),
        cv::FILLED,
        cv::LINE_AA
    );

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

void drawDebugOverlay(cv::Mat& frame, const cv::Rect2f& face) {
    cv::rectangle(frame, clampRect(face, frame.size()), cv::Scalar(40, 220, 120), 2, cv::LINE_AA);
}

void applyOverlay(cv::Mat& frame, const AppConfig& config, const cv::Rect2f& face) {
    if (config.overlay == "hat") {
        drawHatOverlay(frame, face);
    } else if (config.overlay == "glasses") {
        drawGlassesOverlay(frame, face);
    }

    if (config.debug) {
        drawDebugOverlay(frame, face);
    }
}

AppConfig parseArgs(int argc, char** argv) {
    const cv::String keys =
        "{help h usage ?||Show help}"
        "{camera|0|Camera index}"
        "{width|640|Capture width}"
        "{height|480|Capture height}"
        "{fps|30|Requested camera FPS}"
        "{detect-interval|4|Frames to track between detector refreshes}"
        "{overlay|hat|Overlay type: hat or glasses}"
        "{backend|dshow|Camera backend: dshow or any}"
        "{debug|false|Show debug rectangle}"
        "{mirror|true|Mirror the selfie preview}";

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
    config.detectInterval = std::max(0, parser.get<int>("detect-interval"));
    config.overlay = parser.get<std::string>("overlay");
    config.backend = parser.get<std::string>("backend");
    config.debug = parser.get<bool>("debug");
    config.mirror = parser.get<bool>("mirror");

    if (!parser.check()) {
        parser.printErrors();
        std::exit(1);
    }

    if (config.overlay != "hat" && config.overlay != "glasses") {
        std::cerr << "overlay must be 'hat' or 'glasses'\n";
        std::exit(1);
    }

    if (config.backend != "dshow" && config.backend != "any") {
        std::cerr << "backend must be 'dshow' or 'any'\n";
        std::exit(1);
    }

    return config;
}

}  // namespace

int main(int argc, char** argv) {
    cv::setUseOptimized(true);

    const AppConfig config = parseArgs(argc, argv);

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

    std::cout << "Native lightweight filter running.\n";
    std::cout << "Keys: q = quit, space = lock face, c = clear lock, 1 = hat, 2 = glasses, d = debug overlay\n";

    AppConfig runtimeConfig = config;

    while (true) {
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
        if (trackState.hasFace && trackState.framesSinceDetection < runtimeConfig.detectInterval) {
            tracked = updateTrackedFace(trackState.previousGray, gray, trackState);
        }

        if (tracked) {
            trackState.smoothedBox = smoothRect(trackState.smoothedBox, trackState.box, 0.22f);
            ++trackState.framesSinceDetection;
        } else if (trackState.hasFace) {
            trackState.hasFace = false;
            trackState.points.clear();
        }

        trackState.previousGray = gray;

        if (trackState.hasFace) {
            applyOverlay(frame, runtimeConfig, trackState.smoothedBox);
        } else {
            const cv::Rect guideRect = clampRect(guideBox, frame.size());
            cv::rectangle(frame, guideRect, cv::Scalar(60, 220, 255), 2, cv::LINE_AA);
            cv::putText(
                frame,
                "Center face and press SPACE",
                cv::Point(guideRect.x, std::max(30, guideRect.y - 12)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(30, 30, 30),
                3,
                cv::LINE_AA
            );
            cv::putText(
                frame,
                "Center face and press SPACE",
                cv::Point(guideRect.x, std::max(30, guideRect.y - 12)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(245, 245, 245),
                1,
                cv::LINE_AA
            );
        }

        const auto now = Clock::now();
        const double seconds = std::chrono::duration<double>(now - previousFrameTime).count();
        previousFrameTime = now;
        if (seconds > 0.0) {
            const double instantFps = 1.0 / seconds;
            fpsEstimate = fpsEstimate <= 0.0 ? instantFps : (fpsEstimate * 0.90) + (instantFps * 0.10);
        }

        cv::putText(
            frame,
            "native-filter",
            cv::Point(16, 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(20, 20, 20),
            3,
            cv::LINE_AA
        );
        cv::putText(
            frame,
            "native-filter",
            cv::Point(16, 28),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(240, 240, 240),
            1,
            cv::LINE_AA
        );

        const std::string stats = "fps " + std::to_string(static_cast<int>(std::round(fpsEstimate))) +
            "  overlay " + runtimeConfig.overlay;
        cv::putText(
            frame,
            stats,
            cv::Point(16, 54),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(10, 10, 10),
            3,
            cv::LINE_AA
        );
        cv::putText(
            frame,
            stats,
            cv::Point(16, 54),
            cv::FONT_HERSHEY_SIMPLEX,
            0.55,
            cv::Scalar(230, 230, 230),
            1,
            cv::LINE_AA
        );

        cv::imshow("LocalFaceFilter Native", frame);

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
        if (key == '1') {
            runtimeConfig.overlay = "hat";
        }
        if (key == '2') {
            runtimeConfig.overlay = "glasses";
        }
        if (key == 'd') {
            runtimeConfig.debug = !runtimeConfig.debug;
        }
    }

    grabber.stop();
    cv::destroyAllWindows();
    return 0;
}
