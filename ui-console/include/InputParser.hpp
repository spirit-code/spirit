#pragma once
#ifndef INPUTPARSER_H
#define INPUTPARSER_H

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

class InputParser
{
    public:
        InputParser (int &argc, char **argv)
        {
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        
        const std::string& getCmdOption(const std::string &option) const
        {
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end())
                return *itr;
            static const std::string empty_string("");
            return empty_string;
        }
        
        bool cmdOptionExists(const std::string &option) const
        {
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }

    private:
        std::vector <std::string> tokens;
};

#endif