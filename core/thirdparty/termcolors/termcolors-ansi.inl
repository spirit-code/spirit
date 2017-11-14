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

namespace termcolors {

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& operator<<(
            std::basic_ostream<CharT, Traits>& out,
            const foreground_color& c) {
        out << "\033["
            << 30 + static_cast<int>(c.newForegroundColor)
            << 'm';
        return out;
    }

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& operator<<(
            std::basic_ostream<CharT, Traits>& out,
            const background_color& c) {
        out << "\033["
            << 40 + static_cast<int>(c.newBackgroundColor)
            << 'm';
        return out;
    }

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& bright(
            std::basic_ostream<CharT, Traits>& out) {
        out << "\033[1m";
        return out;
    }
    
    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& normal(
            std::basic_ostream<CharT, Traits>& out) {
        out << "\033[22m";
        return out;
    }

    template<typename CharT, class Traits>
        std::basic_ostream<CharT, Traits>& reset(
            std::basic_ostream<CharT, Traits>& out) {
        out << "\033[0m";
        return out;
    }

}

#endif
