template<typename T>
ArrayGather<T>::ArrayGather(const Array<T>& array, const Array<int>& indices)
    : array_(array), indices_(indices) {}

template<typename T>
ArrayGather<T>& ArrayGather<T>::operator+=(const Array<T>& updates) {
    scatter_saver<saver::Increment>(array_, indices_, updates);
    return *this;
}

template<typename T>
ArrayGather<T>& ArrayGather<T>::operator=(const Array<T>& updates) {
    scatter_saver<saver::Assign>(array_, indices_, updates);
    return *this;
}

template<typename T>
ArrayGather<T>& ArrayGather<T>::operator-=(const Array<T>& updates) {
    scatter_saver<saver::Decrement>(array_, indices_, updates);
    return *this;
}
