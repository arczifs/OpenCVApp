#ifndef QPROPERTYWRAPPER_H
#define QPROPERTYWRAPPER_H

#include <functional>
#include <type_traits>
#include <assert.h>

template<class T>
class QPropertyWrapper {

    using Signal = std::function<void()>;
public:

    inline explicit QPropertyWrapper(const T& defaultValue = T{})
        : m_Value(defaultValue)
    {}

    template <typename Class>
    inline explicit QPropertyWrapper(Class *object, void (Class::*method)(), const T& defaultValue = T{})
        : m_Signal([object, method] { (object->*method)(); })
        , m_Value(defaultValue)
    {
        assert(object && method);
    }

    inline T operator()() const { return m_Value; }
    inline void operator()(const T& value) { set(value); }
    inline operator T() { return m_Value; }
    inline const QPropertyWrapper<T> &operator=(const T& value) { set(value); return *this;}

    template<typename Type = T>
    typename std::enable_if<!std::is_pointer<Type>::value, const T*>::type
    inline operator->() const { return &m_value; }

    template<typename Type = T>
    typename std::enable_if<std::is_pointer<Type>::value, const T>::type
    inline operator->() const { return m_value;}

    inline void set(T value)
    {
        if (value == m_Value)
            return;

        m_Value = value;
        m_Signal();
    }

private:
    T m_Value = {};
    Signal m_Signal = {};

};

#endif // QPROPERTYWRAPPER_H
