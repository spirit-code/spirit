#include "VFRendering/FPSCounter.hxx"

namespace VFRendering {
namespace Utilities {
FPSCounter::FPSCounter(int max_n) : m_max_n(max_n) {}

void FPSCounter::tick() {
    if (m_previous_frame_time_point != std::chrono::steady_clock::time_point()) {
        auto previous_duration = std::chrono::steady_clock::now() - m_previous_frame_time_point;
        m_n_frame_duration += previous_duration;
        m_frame_durations.push(previous_duration);
        while (m_frame_durations.size() > unsigned(m_max_n)) {
            m_n_frame_duration -= m_frame_durations.front();
            m_frame_durations.pop();
        }
    }
    m_previous_frame_time_point = std::chrono::steady_clock::now();
}

float FPSCounter::getFramerate() const {
    if (m_n_frame_duration.count() == 0) {
        return 0;
    }
    return m_frame_durations.size() / m_n_frame_duration.count();
}
}
}
