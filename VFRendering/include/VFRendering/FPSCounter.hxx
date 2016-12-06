#ifndef VFRENDERING_FPS_COUNTER_HXX
#define VFRENDERING_FPS_COUNTER_HXX

#include <chrono>
#include <queue>

namespace VFRendering {
namespace Utilities {
class FPSCounter {
public:
    FPSCounter(int max_n=60);
    void tick();
    float getFramerate() const;

private:
    int m_max_n;
    std::chrono::duration<float> m_n_frame_duration = std::chrono::duration<float>::zero();
    std::chrono::steady_clock::time_point m_previous_frame_time_point;
    std::queue<std::chrono::duration<float>> m_frame_durations;
};
}
}
#endif
