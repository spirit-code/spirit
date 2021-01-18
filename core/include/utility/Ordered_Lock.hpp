#pragma once
#ifndef SPIRIT_ORDERED_LOCK_HPP
#define SPIRIT_ORDERED_LOCK_HPP

#include <condition_variable>
#include <mutex>
#include <queue>

class Ordered_Lock
{
    std::queue<std::condition_variable> condition_;
    std::mutex condition_mutex_;
    bool locked_;

public:
    Ordered_Lock() : locked_( false ){};

    void lock()
    {
        std::unique_lock<std::mutex> condition_lock( condition_mutex_ );
        if( locked_ )
        {
            condition_.emplace();
            condition_.back().wait( condition_lock, [&] { return !locked_; } );
            condition_.pop();
        }
        else
            locked_ = true;
    }

    void unlock()
    {
        std::unique_lock<std::mutex> condition_lock( condition_mutex_ );
        locked_ = false;
        if( !condition_.empty() )
            condition_.front().notify_one();
    }
};

#endif