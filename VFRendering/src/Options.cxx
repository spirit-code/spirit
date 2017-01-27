#include <VFRendering/Options.hxx>

namespace VFRendering {
namespace Utilities {

Options::Options() {
}

std::vector<int> Options::keys() const {
    std::vector<int> keys;
    for (const auto& index_option : m_options) {
        keys.push_back(index_option.first);
    }
    return keys;
}

std::vector<int> Options::update(const Options &other) {
    std::vector<int> updatedOptions;
    for (const auto& index_option : other.m_options) {
        const auto& index = index_option.first;
        const auto& option = index_option.second;
        m_options[index] = option;
        updatedOptions.push_back(index);
    }
    return updatedOptions;
}

}
}
