#pragma once
#ifndef SPIRIT_CORE_UTILITY_EXECUTION_HPP
#define SPIRIT_CORE_UTILITY_EXECUTION_HPP

#include <thread>                   

#include <utility/Exception.hpp>


#ifdef SPIRIT_USE_STDEXEC

// #include <stdexec/execution.hpp>
// #include <exec/static_thread_pool.hpp>

#include <experimental/stdexec/execution.hpp>
#include <experimental/exec/static_thread_pool.hpp>

// #include <nvexec/stream_context.cuh>
// #include <nvexec/multi_gpu_context.cuh>


#endif


namespace Execution {


struct Void_Schedule {};

struct Void_Context {
    static Void_Schedule get_scheduler() noexcept { return Void_Schedule{}; }
};



struct Execution_Shape
{
    int threads = 1;
};



#ifdef SPIRIT_USE_STDEXEC

class Compute_Resource {
public:
    friend class Context;

    explicit
    Compute_Resource (int num_threads = std::thread::hardware_concurrency()):
        shape_{ .threads = num_threads },
        thread_pool_(num_threads)
    {}

private:
    Execution_Shape shape_;
    exec::static_thread_pool thread_pool_;
    // int gpu_count_;
    // nvexec::stream_context stream_context_; 
};


#else   // SPIRIT_USE_STDEXEC

class Compute_Resource {
public:
    friend class Context;

    explicit
    Compute_Resource (int = 0) {}

private:
    inline static constexpr Execution_Shape shape_ {};
    Void_Context thread_pool_;
    Void_Context stream_context_;
};

#endif  // SPIRIT_USE_STDEXEC




/* 
 * Reference to compute resources that can be stored / passed by value
 */
class Context
{
public:
    Context () = default;

    Context (Compute_Resource& res) noexcept: res_{&res} {}


    [[nodiscard]]
    bool is_engaged () const noexcept {
        return static_cast<bool>(res_);
    }

    [[nodiscard]]
    auto get_scheduler () { 
        if (not is_engaged()) {
            spirit_throw(
                Utility::Exception_Classifier::Standard_Exception,
                Utility::Log_Level::Error,
                "Tried to used non-engaged execution context");
        }
        return res_->thread_pool_.get_scheduler();
    }

    [[nodiscard]]
    Execution_Shape resource_shape () const noexcept { 
        return static_cast<bool>(res_) ? res_->shape_ : Execution_Shape{};
    }

            
private:
    Compute_Resource* res_ = nullptr;

};



}


#endif

