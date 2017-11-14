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

#ifndef TERM_COLORS_H
#define TERM_COLORS_H

#include <ostream>

/**
 * @namespace termcolors
 * Holds all symbols in term-colors.
 */
namespace termcolors {
    /**
     * Colors, according to ANSI color escapes.  Bright variants are
     * listed in parentheses when they differ in hue.
     */
    enum class color {
        reset   = 9,            /**< Reset to previous color */
        black   = 0,            /**< Black (Dark Gray) */
        red     = 1,            /**< Red */
        green   = 2,            /**< Green */
        yellow  = 3,            /**< Yellow */
        blue    = 4,            /**< Blue */
        magenta = 5,            /**< Magenta */
        cyan    = 6,            /**< Cyan */
        white   = 7,            /**< White (Pure White) */
    };

    /**
     * A manipulator to set the foreground color of the terminal.
     *
     *     std::cout << foreground_color(color::black) << "Hi";
     */
    struct foreground_color {
        /**
         * Create a new manipulator object that will change the
         * terminal color to the desired foreground color.
         */
        explicit foreground_color(color c) : newForegroundColor(c) {}
        color newForegroundColor; /**< The text color to set */
    };
    /**
     * Apply the foreground_color manipulator.
     */
    template<typename CharT, class Traits = std::char_traits<CharT>>
        std::basic_ostream<CharT, Traits>& operator<<(
            std::basic_ostream<CharT, Traits>&,
            const foreground_color&);

    /**
     * A manipulator to set the background color of the terminal.
     *
     *     std::cout << background_color(color::black) << "Hi";
     */
    struct background_color {
        /**
         * Create a new manipulator object that will change the
         * terminal color to the desired background color.
         */
        explicit background_color(color c) : newBackgroundColor(c) {}
        color newBackgroundColor; /**< The background to set */
    };
    /**
     * Apply the background color manipulator.
     */
    template<typename CharT, class Traits = std::char_traits<CharT>>
        std::basic_ostream<CharT, Traits>& operator<<(
            std::basic_ostream<CharT, Traits>&,
            const background_color&);

    /**
     * Use the bright variant for foreground and background colors.
     */
    template<typename CharT, class Traits = std::char_traits<CharT>>
        std::basic_ostream<CharT, Traits>& bright(
            std::basic_ostream<CharT, Traits>&);
    /**
     * Use the normal variant for foreground and background colors.
     */
    template<typename CharT, class Traits = std::char_traits<CharT>>
        std::basic_ostream<CharT, Traits>& normal(
            std::basic_ostream<CharT, Traits>&);

    /**
     * Reset all traits to default.
     */
    template<typename CharT, class Traits = std::char_traits<CharT>>
        std::basic_ostream<CharT, Traits>& reset(
            std::basic_ostream<CharT, Traits>&);
}

// Implementation
#if defined(_WIN32)
#    include "termcolors-win.inl"
#else
#    include "termcolors-ansi.inl"
#endif

#endif
