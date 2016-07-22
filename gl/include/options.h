#ifndef OPTIONS_H
#define OPTIONS_H
#include <vector>
#include <unordered_map>
#include <memory>


template<typename T>
class Options {
private:
  template<int i>
  struct Type;
  
public:
  template<int index>
  typename Type<index>::type get() const;
  
  template<int index>
  bool has() const;
  
  template<int index>
  void set(typename Type<index>::type value);
  
  std::vector<int> update(const Options<T> &other);
  
private:
  template<int i>
  struct Option;
  
  template<int i>
  struct Type {
    typedef decltype(Option<i>().default_value) type;
  };
  
  struct IStorableOption {
    virtual ~IStorableOption() {}
  };
  
  template<int i>
  struct StorableOption : public IStorableOption {
    StorableOption(const typename Type<i>::type& value) : _value(value) {}
    typename Type<i>::type _value;
  };
  
  std::unordered_map<int, std::shared_ptr<IStorableOption>> _options;
  
  template<typename U>
  static inline const U& asConstRef(const U& value) {
    return value;
  }
};

template<typename T>
std::vector<int> Options<T>::update(const Options<T> &other) {
  std::vector<int> updatedOptions;
  for (auto it = other._options.cbegin(); it != other._options.cend(); it++) {
    _options[it->first] = it->second;
    updatedOptions.push_back(it->first);
  }
  return updatedOptions;
}

template<typename T>
template<int index>
typename Options<T>::template Type<index>::type Options<T>::get() const {
  if (has<index>()) {
    if (auto storableOption = static_cast<StorableOption<index> *>(_options.at(index).get())) {
      return storableOption->_value;
    }
  }
  return asConstRef(Option<index>().default_value);
}

template<typename T>
template<int index>
bool Options<T>::has() const {
  return (_options.find(index) != _options.end());
}

template<typename T>
template<int index>
void Options<T>::set(typename Type<index>::type value) {
  _options[index] = std::make_shared<StorableOption<index>>(value);
}
  
#endif
