#ifndef VECTORFIELDRENDERING_OPTIONS_HXX
#define VECTORFIELDRENDERING_OPTIONS_HXX

#include <vector>
#include <unordered_map>
#include <memory>
#include <iostream>

namespace VFRendering {
namespace Utilities {

class Options {
private:
    template<int i>
    struct Option;

public:

    template<int i>
    struct Type {
        typedef decltype(Option<i>().default_value) type;
    };

    Options();

    template<int index>
    static Options withOption(typename Type<index>::type value);

    template<int index>
    typename Type<index>::type get() const;

    template<int index>
    bool has() const;

    template<int index>
    void set(typename Type<index>::type value);

    template<int index>
    void clear();

    std::vector<int> keys() const;

    std::vector<int> update(const Options &other);

private:
    struct IStorableOption {
        virtual ~IStorableOption() {}
    };

    template<int i>
    struct StorableOption : public IStorableOption {
        StorableOption(const typename Type<i>::type& value) : m_value(value) {}
        typename Type<i>::type m_value;
    };

    std::unordered_map<int, std::shared_ptr<IStorableOption>> m_options;
};


template<int index>
Options Options::withOption(typename Type<index>::type value) {
    Options options;
    options.set<index>(value);
    return options;
}

template<int index>
typename Options::template Type<index>::type Options::get() const {
    if (has<index>()) {
        if (auto storableOption = static_cast<StorableOption<index> *>(m_options.at(index).get())) {
            return storableOption->m_value;
        }
    }
    return Option<index>().default_value;
}

template<int index>
bool Options::has() const {
    return (m_options.find(index) != m_options.end());
}

template<int index>
void Options::set(typename Type<index>::type value) {
    m_options[index] = std::make_shared<StorableOption<index>>(value);
}

template<int index>
void Options::clear() {
    m_options.erase(index);
}

}
}

#endif
