#pragma once
#ifndef SPIRIT_ORDERED_LOCK_HPP
#define SPIRIT_ORDERED_LOCK_HPP

#include <condition_variable>
#include <mutex>
#include <queue>

class Ordered_Lock
{
    std::queue<std::condition_variable> cvar;
    std::mutex                          cvar_lock;
    bool                                locked;

public:
    Ordered_Lock() : locked(false) {};

    void lock()
    {
        std::unique_lock<std::mutex> acquire(cvar_lock);
        if( locked )
        {
            cvar.emplace();
            cvar.back().wait(acquire);
        }
        else
            locked = true;
    }

    void unlock()
    {
        std::unique_lock<std::mutex> acquire(cvar_lock);
        if( cvar.empty() )
            locked = false;
        else
        {
            cvar.front().notify_one();
            cvar.pop();
        }
    }
};

#endif