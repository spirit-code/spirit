// -*- c++ -*-
/* This file is a part of term-colors.
 * Copyright (C) 2012, Patrick M. Niedzielski.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *  1. Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *
 *  3. The names of its contributors may not be used to endorse or
 *     promote products derived from this software without specific
 *     prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef TERM_COLORS_ANSI_INL
#define TERM_COLORS_ANSI_INL

#include <ostream>

#include <windows.h>

namespace termcolors {

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& operator<<(
            std::basic_ostream<CharT, Traits>& out,
            const foreground_color& c) {
        // Do we want to do something different for STD_ERROR_HANDLE
        // here?  They redirect to the same buffer.  It mightly only
        // appear in corner cases.
        HANDLE out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

        // Get current info for finding current background.
        PCONSOLE_SCREEN_BUFFER_INFO info;
        WORD attributes;
        if (0 == GetConsoleScreenBufferInfo(out_handle, info)) {
            // Error.  Don't do anything but set the error flag.
            out.setstate(std::ios_base::badbit);
            return out;
        }
        attributes = info->wAttributes;

        // Clear out foreground attributes;
        attributes &= !(FOREGROUND_BLUE | FOREGROUND_GREEN |
                        FOREGROUND_RED);

        // Turn them on based on our colors.
        if (c.newForegroundColor == color::red     ||
            c.newForegroundColor == color::yellow  ||
            c.newForegroundColor == color::magenta ||
            c.newForegroundColor == color::white) {
            attributes |= FOREGROUND_RED;
        }
        if (c.newForegroundColor == color::green   ||
            c.newForegroundColor == color::yellow  ||
            c.newForegroundColor == color::cyan    ||
            c.newForegroundColor == color::white) {
            attributes |= FOREGROUND_GREEN;
        }
        if (c.newForegroundColor == color::blue    ||
            c.newForegroundColor == color::magenta ||
            c.newForegroundColor == color::cyan    ||
            c.newForegroundColor == color::white) {
            attributes |= FOREGROUND_BLUE;
        }

        // Set it.
        if (0 == SetConsoleTextAttribute(out_handle, attributes)) {
            // Error.  Set the error flag.
            out.setstate(std::ios_base::badbit);
        }

        return out;
    }

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& operator<<(
            std::basic_ostream<CharT, Traits>& out,
            const background_color& c) {
        // Do we want to do something different for STD_ERROR_HANDLE
        // here?  They redirect to the same buffer.  It mightly only
        // appear in corner cases.
        HANDLE out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

        // Get current info for finding current background.
        PCONSOLE_SCREEN_BUFFER_INFO info;
        WORD attributes;
        if (0 == GetConsoleScreenBufferInfo(out_handle, info)) {
            // Error.  Don't do anything but set the error flag.
            out.setstate(std::ios_base::badbit);
            return out;
        }
        attributes = info->wAttributes;

        // Clear out foreground attributes;
        attributes &= !(BACKGROUND_BLUE | BACKGROUND_GREEN |
                        BACKGROUND_RED);

        // Turn them on based on our colors.
        if (c.newBackgroundColor == color::red     ||
            c.newBackgroundColor == color::yellow  ||
            c.newBackgroundColor == color::magenta ||
            c.newBackgroundColor == color::white) {
            attributes |= BACKGROUND_RED;
        }
        if (c.newBackgroundColor == color::green   ||
            c.newBackgroundColor == color::yellow  ||
            c.newBackgroundColor == color::cyan    ||
            c.newBackgroundColor == color::white) {
            attributes |= BACKGROUND_GREEN;
        }
        if (c.newBackgroundColor == color::blue    ||
            c.newBackgroundColor == color::magenta ||
            c.newBackgroundColor == color::cyan    ||
            c.newBackgroundColor == color::white) {
            attributes |= BACKGROUND_BLUE;
        }

        // Set it.
        if (0 == SetConsoleTextAttribute(out_handle, attributes)) {
            // Error.  Set the error flag.
            out.setstate(std::ios_base::badbit);
        }

        return out;
    }

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& bright(
            std::basic_ostream<CharT, Traits>& out) {
        // Do we want to do something different for STD_ERROR_HANDLE
        // here?  They redirect to the same buffer.  It mightly only
        // appear in corner cases.
        HANDLE out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

        // Get current info for finding current background.
        PCONSOLE_SCREEN_BUFFER_INFO info;
        WORD attributes;
        if (0 == GetConsoleScreenBufferInfo(out_handle, info)) {
            // Error.  Don't do anything but set the error flag.
            out.setstate(std::ios_base::badbit);
            return out;
        }
        attributes = info->wAttributes | FOREGROUND_INTENSITY
            | BACKGROUND_INTENSITY;

        // Set it.
        if (0 == SetConsoleTextAttribute(out_handle, attributes)) {
            // Error.  Set the error flag.
            out.setstate(std::ios_base::badbit);
        }

        return out;
    }
    
    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& normal(
            std::basic_ostream<CharT, Traits>& out) {
        // Do we want to do something different for STD_ERROR_HANDLE
        // here?  They redirect to the same buffer.  It mightly only
        // appear in corner cases.
        HANDLE out_handle = GetStdHandle(STD_OUTPUT_HANDLE);

        // Get current info for finding current background.
        PCONSOLE_SCREEN_BUFFER_INFO info;
        WORD attributes;
        if (0 == GetConsoleScreenBufferInfo(out_handle, info)) {
            // Error.  Don't do anything but set the error flag.
            out.setstate(std::ios_base::badbit);
            return out;
        }
        attributes = info->wAttributes | !FOREGROUND_INTENSITY
            | !BACKGROUND_INTENSITY;

        // Set it.
        if (0 == SetConsoleTextAttribute(out_handle, attributes)) {
            // Error.  Set the error flag.
            out.setstate(std::ios_base::badbit);
        }

        return out;
    }

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& reset(
            std::basic_ostream<CharT, Traits>& out) {
        out << normal
            << foreground_color(color::reset)
            << background_color(color::reset);
        return out;
    }

}

#endif
