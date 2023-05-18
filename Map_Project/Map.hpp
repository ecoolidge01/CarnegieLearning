#ifndef MAP_HPP
#define MAP_HPP
#include <stdlib.h>
#include <utility>
#include <stdexcept>
#include <stdio.h>
#include <ostream>
#include <iostream>

namespace cs440{
    template <typename Key_T, typename Mapped_T> class Map{
        typedef std::pair<const Key_T, Mapped_T> ValueType;
        friend bool operator==(const Map & map1, const Map & map2){
            if(map1.count != map2.count || !(map1.root->data == map2.root->data)){
                return false;
            }
            bool equal = true;
            ConstIterator iter1(iter1);
            ConstIterator iter2(iter2);
            iter1 = map1.begin();
            iter2 = map2.begin();

            while(iter1 != map1.end() && equal){
                if(!(*iter1 == *iter2)){
                    equal = false;
                }
                iter1++;
                iter2++;
            }
            return equal;
        }
        friend bool operator!=(const Map & map1, const Map & map2){
            return !(map1 == map2);
        }
        friend bool operator<(const Map & map1, const Map & map2){
            if(map1.count > map2.count){
                return false;
            }
            if(map1.count < map2.count){
                return true;
            }
            bool lessThan = true;
            ConstIterator iter1(iter1);
            ConstIterator iter2(iter2);
            iter1 = map1.begin();
            iter2 = map2.begin();

            while(iter1 != map1.end() && lessThan){
                if((*iter1 == *iter2) || (*iter2 < *iter1)){
                    lessThan = false;
                }
                iter1++;
                iter2++;
            }
            return lessThan;
        }
        private:
            class Node{
                public:
                    ValueType data;
                    Node * left;
                    Node * right;
                    Node * prev;
                    int height;
                    Node() : data(NULL, NULL){
                        left = nullptr;
                        right = nullptr;
                        prev = nullptr;
                        height = 1;
                    }
                    Node(ValueType input) : data(input.first, input.second){
                        left = nullptr;
                        right = nullptr;
                        prev = nullptr;
                        height = 1;
                    }
                    void setPrev(Node * node){
                        prev = node;
                    }
            };
            int count;
            Node * root;
        public:
            class ConstIterator;
            class Iterator{
                friend class Map;
                friend class ConstIterator;
                friend bool operator==(const Iterator & iter1, const Iterator & iter2){
                    if(iter1.current == iter2.current){
                        return true;
                    }
                    return false;
                }
                friend bool operator!=(const Iterator & iter1, const Iterator & iter2){
                    return !(iter1 == iter2);
                }
                friend bool operator==(const Iterator & iter1, const ConstIterator & iter2){
                    if(iter1.current == iter2.current){
                        return true;
                    }
                    return false;
                }
                friend bool operator==(const ConstIterator & iter1, const Iterator & iter2){
                    if(iter1.current == iter2.current){
                        return true;
                    }
                    return false;
                }
                friend bool operator!=(const Iterator & iter1, const ConstIterator & iter2){
                    return !(iter1 == iter2);
                }
                friend bool operator!=(const ConstIterator & iter1, const Iterator & iter2){
                    return !(iter1 == iter2);
                }
                private:
                    Node * current;
                    Node * root;
                    bool end;
                    bool begin;
                public:
                    Iterator(const Iterator & Iter){
                        current = Iter.current;
                        end = false;
                        begin = false;
                        root = nullptr;
                    }
                    ~Iterator(){};
                    Node * getCurrent(){
                        return current;
                    }
                    Iterator& operator=(const Iterator & Iter){
                        current = Iter.current;
                        end = Iter.end;
                        begin = Iter.begin;
                        root = Iter.root;
                        return *this;
                    }
                    Iterator &operator++(){
                        if(!end){
                            //current is right child
                            if(current->prev == nullptr || current->prev->right == current){
                                if(current->right == nullptr && current->prev == nullptr){
                                    current = nullptr;
                                    end = true;
                                }
                                else if(current->right == nullptr){
                                    Node * temp = current->prev;
                                    while(temp->data.first < current->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(current->data.first < temp->data.first){
                                            current = temp;
                                    }
                                    else{
                                        current = nullptr;
                                        end = true;
                                    }
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->right == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        return *this;
                    }
                    Iterator operator++(int i){
                        Iterator ret(ret);
                        if(!end){
                            ret.current = current;
                            ret.end = end;
                            ret.begin = begin;
                            ret.root = root;
                            //current is right child
                            if(current->prev == nullptr || current->prev->right == current){
                                if(current->right == nullptr && current->prev == nullptr){
                                    current = nullptr;
                                    end = true;
                                }
                                else if(current->right == nullptr){
                                    Node * temp = current->prev;
                                    while(temp->data.first < current->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(current->data.first < temp->data.first){
                                        current = temp;
                                    }
                                    else{
                                        current = nullptr;
                                        end = true;
                                    }
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->right == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        return ret;
                    }
                    Iterator &operator--(){
                        if(!end && !begin){
                            //current is left child
                            if(current->prev == nullptr || current->prev->left == current){
                                if(current->left == nullptr && current->prev == nullptr){
                                    begin = true;
                                }
                                else if(current->left == nullptr){
                                    Node * temp = current->prev;
                                    while(current->data.first < temp->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(temp->data.first < current->data.first){
                                        current = temp;
                                    }
                                    else{
                                        begin = true;
                                    }

                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->left == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        else if(!begin){
                            Node * temp = root;
                            while(temp->right != nullptr){
                                temp = temp->right;
                            }
                            current = temp;
                            end = false;
                        }
                        return *this;
                    }
                    Iterator operator--(int i){
                        Iterator ret(ret);
                        ret.current = current;
                        ret.end = end;
                        ret.root = root;
                        ret.begin = begin;
                        if(!end && !begin){
                            //current is left child
                            if(current->prev == nullptr || current->prev->left == current){
                                if(current->left == nullptr && current->prev == nullptr){
                                    begin = true;
                                }
                                else if(current->left == nullptr){
                                    Node * temp = current->prev;
                                    while(current->data.first < temp->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(temp->data.first < current->data.first){
                                        current = temp;
                                    }
                                    else{
                                        begin = true;
                                    }

                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->left == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        else if(!begin){
                            Node * temp = root;
                            while(temp->right != nullptr){
                                temp = temp->right;
                            }
                            current = temp;
                            end = false;
                        }
                        return ret;
                    }
                    ValueType &operator*() const{
                        return this->current->data;
                    }
                    ValueType *operator->() const{
                        return &(this->current->data);
                    }
                    // bool operator==(const ConstIterator & iter){
                    // if(this->current == iter.current){
                    //     return true;
                    // }
                    //     return false;
                    // }
                    // bool operator!=(const ConstIterator & iter){
                    //     return !(*this == iter);
                    // }
            };
            class ConstIterator{
                friend class Map;
                friend class Iterator;
                private:
                    Node * current;
                    bool end;
                    bool begin;
                    Node * root;
                public:
                    ConstIterator(const ConstIterator & constIter){
                        current = constIter.current;
                        end = false;
                        begin = false;
                        root = nullptr;
                    }
                    ConstIterator(const Iterator & Iter){
                        current = Iter.current;
                        end = Iter.end;
                        begin = Iter.begin;
                        root = Iter.root;
                    }
                    Node * getCurrent(){
                        return current;
                    }
                    ~ConstIterator(){};
                    ConstIterator& operator=(const ConstIterator & constIter){
                        current = constIter.current;
                        end = constIter.end;
                        begin = constIter.begin;
                        root = constIter.root;
                        return *this;
                    }
                    ConstIterator &operator++(){
                        if(!end){
                            //current is right child
                            if(current->prev == nullptr || current->prev->right == current){
                                if(current->right == nullptr && current->prev == nullptr){
                                    current = nullptr;
                                    end = true;
                                }
                                else if(current->right == nullptr){
                                    Node * temp = current->prev;
                                    while(temp->data.first < current->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(current->data.first < temp->data.first){
                                            current = temp;
                                    }
                                    else{
                                        current = nullptr;
                                        end = true;
                                    }
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->right == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        return *this;
                    }
                    ConstIterator operator++(int){
                       ConstIterator ret(ret);
                        if(!end){
                            ret.current = current;
                            ret.end = end;
                            ret.begin = begin;
                            ret.root = root;
                            //current is right child
                            if(current->prev == nullptr || current->prev->right == current){
                                if(current->right == nullptr && current->prev == nullptr){
                                    current = nullptr;
                                    end = true;
                                }
                                else if(current->right == nullptr){
                                    Node * temp = current->prev;
                                    while(temp->data.first < current->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(current->data.first < temp->data.first){
                                        current = temp;
                                    }
                                    else{
                                        current = nullptr;
                                        end = true;
                                    }
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->right == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        return ret;
                    }
                    ConstIterator &operator--(){
                        if(!end && !begin){
                            //current is left child
                            if(current->prev == nullptr || current->prev->left == current){
                                if(current->left == nullptr && current->prev == nullptr){
                                    begin = true;
                                }
                                else if(current->left == nullptr){
                                    Node * temp = current->prev;
                                    while(current->data.first < temp->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(temp->data.first < current->data.first){
                                        current = temp;
                                    }
                                    else{
                                        begin = true;
                                    }

                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->left == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        else if(!begin){
                            Node * temp = root;
                            while(temp->right != nullptr){
                                temp = temp->right;
                            }
                            current = temp;
                            end = false;
                        }
                        return *this;
                    }
                    ConstIterator operator--(int){
                        Iterator ret(ret);
                        ret.current = current;
                        ret.end = end;
                        ret.root = root;
                        ret.begin = begin;
                        if(!end && !begin){
                            //current is left child
                            if(current->prev == nullptr || current->prev->left == current){
                                if(current->left == nullptr && current->prev == nullptr){
                                    begin = true;
                                }
                                else if(current->left == nullptr){
                                    Node * temp = current->prev;
                                    while(current->data.first < temp->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(temp->data.first < current->data.first){
                                        current = temp;
                                    }
                                    else{
                                        begin = true;
                                    }

                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->left == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        else if(!begin){
                            Node * temp = root;
                            while(temp->right != nullptr){
                                temp = temp->right;
                            }
                            current = temp;
                            end = false;
                        }
                        return ret;
                    }
                const ValueType &operator*() const{
                    return this->current->data;
                }
                const ValueType *operator->() const{
                    return &(this->current->data);
                }
                bool operator==(const ConstIterator & iter){
                    if(iter.current == this->current){
                        return true;
                    }
                    return false;
                }
                bool operator!=(const ConstIterator & iter){
                    return !(*this == iter);
                }
                // bool operator==(const Iterator & iter){
                //     if(iter.current == this->current){
                //         return true;
                //     }
                //     return false;
                // }
                // bool operator!=(const Iterator & iter){
                //     return !(*this == iter);
                // }
            };
            class ReverseIterator{
                friend class Map;
                friend bool operator==(const ReverseIterator & iter1, const ReverseIterator & iter2){
                    if(iter1.current == iter2.current){
                        return true;
                    }
                    return false;
                }
                friend bool operator!=(const ReverseIterator & iter1, const ReverseIterator & iter2){
                    return !(iter1 == iter2);
                }
                private:
                    Node * current;
                    bool begin;
                    bool end;
                    Node * root;
                public:
                    ReverseIterator(const ReverseIterator & revIter){
                        current = revIter.current;
                        begin = false;
                        end = false;
                        root = nullptr;
                    }
                    Node * getCurrent(){
                        return current;
                    }
                    ~ReverseIterator(){};
                    ReverseIterator& operator=(const ReverseIterator & revIter){
                        current = revIter.current;
                        begin = revIter.begin;
                        end = revIter.end;
                        root = revIter.root;
                        return *this;
                    }
                    ReverseIterator &operator++(){
                        if(!end){
                            //current is left child
                            if(current->prev == nullptr || current->prev->left == current){
                                if(current->left == nullptr && current->prev == nullptr){
                                    current = nullptr;
                                    end = true;
                                }
                                else if(current->left == nullptr){
                                    Node * temp = current->prev;
                                    while(current->data.first < temp->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(temp->data.first < current->data.first){
                                        current = temp;
                                    }
                                    else{
                                        current = nullptr;
                                        end = true;
                                    }

                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->left == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        return *this;
                    }
                    ReverseIterator operator++(int){
                        ReverseIterator ret(ret);
                        ret.current = current;
                        ret.root = root;
                        ret.end = end;
                        if(!end){
                            //current is left child
                            if(current->prev == nullptr || current->prev->left == current){
                                if(current->left == nullptr && current->prev == nullptr){
                                    current = nullptr;
                                    end = true;
                                }
                                else if(current->left == nullptr){
                                    Node * temp = current->prev;
                                    while(current->data.first < temp->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(temp->data.first < current->data.first){
                                        current = temp;
                                    }
                                    else{
                                        current = nullptr;
                                        end = true;
                                    }

                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->left == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->left;
                                    while(temp->right != nullptr){
                                        temp = temp->right;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        return ret;
                    }
                    ReverseIterator &operator--(){
                        if(!end && !begin){
                            //current is right child
                            if(current->prev == nullptr || current->prev->right == current){
                                if(current->right == nullptr && current->prev == nullptr){
                                    begin = true;
                                }
                                else if(current->right == nullptr){
                                    Node * temp = current->prev;
                                    while(temp->data.first < current->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(current->data.first < temp->data.first){
                                        current = temp;
                                    }
                                    else{
                                        begin = true;
                                    }
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->right == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        else if(!begin){
                            Node * temp = root;
                            while(temp->left != nullptr){
                                temp = temp->left;
                            }
                            current = temp;
                            end = false;
                        }
                        return *this;
                    }
                    ReverseIterator operator--(int){
                        ReverseIterator ret(ret);
                        if(!end && !begin){
                            ret.current = current;
                            ret.end = end;
                            ret.begin = begin;
                            ret.root = root;
                            //current is right child
                            if(current->prev == nullptr || current->prev->right == current){
                                if(current->right == nullptr && current->prev == nullptr){
                                    begin = true;
                                }
                                else if(current->right == nullptr){
                                    Node * temp = current->prev;
                                    while(temp->data.first < current->data.first && temp->prev != nullptr){
                                        temp = temp->prev;
                                    }
                                    if(current->data.first < temp->data.first){
                                        current = temp;
                                    }
                                    else{
                                        begin = true;
                                    }
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                            else{
                                if(current->right == nullptr){
                                    current = current->prev;
                                }
                                else{
                                    Node * temp = current->right;
                                    while(temp->left != nullptr){
                                        temp = temp->left;
                                    }
                                    current = temp;
                                }
                            }
                        }
                        else if(!begin){
                            Node * temp = root;
                            while(temp->left != nullptr){
                                temp = temp->left;
                            }
                            current = temp;
                            end = false;
                        }
                        return ret;
                    }
                ValueType &operator*() const{
                    return this->current->data;
                }
                ValueType *operator->() const{
                    return &(this->current->data);
                }
            };
            Iterator begin(){
                Iterator ret(ret);
                if(root != nullptr){
                    Node * current = root;
                    while(current->left != nullptr){
                        current = current->left;
                    }
                    ret.current = current;
                    ret.begin = true;
                    ret.root = root;
                    return ret;
                }
                ret.current = nullptr;
                return ret;
            }
            Iterator end(){
                Iterator ret(ret);
                ret.current = nullptr;
                ret.end = true;
                ret.root = root;
                return ret;
            }
            ConstIterator begin() const{
                Iterator ret(ret);
                if(root != nullptr){
                    Node * current = root;
                    while(current->left != nullptr){
                        current = current->left;
                    }
                    ret.current = current;
                    ret.begin = true;
                    ret.root = root;
                    return ret;
                }
                ret.current = nullptr;
                return ret;
            }
            ConstIterator end() const{
                Iterator ret(ret);
                ret.current = nullptr;
                ret.end = true;
                ret.root = root;
                return ret;
            }
            ReverseIterator rbegin(){
                ReverseIterator ret(ret);
                if(root != nullptr){
                    Node * current = root;
                    while(current->right != nullptr){
                        current = current->right;
                    }
                    ret.current = current;
                    ret.begin = true;
                    ret.root = root;
                    return ret;
                }
                ret.current = nullptr;
                return ret;
            }
            ReverseIterator rend(){
                ReverseIterator ret(ret);
                ret.current = nullptr;
                ret.end = true;
                ret.root = root;
                return ret;
            }
            Map(){
                root = nullptr;
                count = 0;
            }
            Map(const Map & input){
                deepCopy(input.root, nullptr, true);
            }
            Node * deepCopy(Node * current, Node * previous, bool isRoot){
                if(current != nullptr){
                    Node * node = new Node(current->data);
                    if(isRoot){
                        root = node;
                        count = 1;
                    }
                    else{
                        node->prev = previous;
                        count++;
                    }
                    node->left = deepCopy(current->left, node, false);
                    node->right = deepCopy(current->right, node, false);
                    return node;
                }
                return nullptr;
            }
            Map &operator=(const Map & input){
                //Node * previous;
                //bool isRoot = false;
                deepCopy(input.root, nullptr, true);
                return *this;
            }
            size_t size() const{
                return count;
            }
            bool empty() const{
                if(count == 0){
                    return true;
                }
                return false;
            }

            //Rotation Functions

            Node * leftLeft(Node * n1, Node * n2){
                Node * n2Left = n2->left;
                n1->right = n2Left;
                if(n2Left != nullptr){
                    n2Left->prev = n1;
                }
                n2->left = n1;
                n1->prev = n2;

                return n2;                                            
            }

            Node * rightRight(Node * n1, Node * n2){
                Node * n2Right = n2->right;
                n1->left = n2Right;
                if(n2Right != nullptr){
                    n2Right->prev = n1;
                }
                n2->right = n1;
                n1->prev = n2;

                return n2;
            }

            Node * leftRight(Node * n1, Node * n2){
                Node * n2Right = n2->right;
                Node * n2RightLeft = n2->right->left;
                n2->right = n2RightLeft;
                if(n2RightLeft != nullptr){
                    n2RightLeft->prev = n2;
                }
                n2Right->left = n2;
                n2->prev = n2Right;
                Node * n2RightRight = nullptr;
                n2RightRight = n2->right->right;
                n1->left = n2RightRight;
                if(n2RightRight != nullptr){
                    n2RightRight->prev = n1;
                }
                n2Right->right = n1;
                n1->prev = n2;
                return n2Right;
            }

            Node * rightLeft(Node * n1, Node * n2){
                Node * n2Left = n2->left;
                Node * n2LeftRight = n2->left->right;
                n2->left = n2LeftRight;
                if(n2LeftRight != nullptr){
                    n2LeftRight->prev = n2;
                }
                n2Left->right = n2;
                n2->prev = n2Left;
                Node * n2LeftLeft = n2->left->left;
                n1->right = n2LeftLeft;
                if(n2LeftLeft != nullptr){
                    n2LeftLeft->prev = n1;
                }
                n2Left->left = n1;
                n1->prev = n2;
                return n2Left;
            }
            
            int balanceFactor(Node * node){
                if(node->left == nullptr && node->right == nullptr){
                    return 0;
                }
                else if(node->left == nullptr && node->right != nullptr){
                    return node->right->height;
                }
                else if(node->left != nullptr && node->right == nullptr){
                    return -1 * (node->left->height);
                }
                else{
                    return node->right->height - node->left->height;
                }
            }

            void balanceTree(Node * node){
                Node * previous = nullptr;
                Node * newRoot = nullptr;
                Node * newPrevious = nullptr;
                for(previous = node->prev; previous != nullptr; previous = node->prev){
                    if(node == previous->right){
                        if(balanceFactor(previous) > 0){
                            newPrevious = previous->prev;
                            if(balanceFactor(node) < 0){
                                newRoot = rightLeft(previous, node);
                                updateAllheights(root);
                            }
                            else{
                                newRoot = leftLeft(previous, node);
                                updateAllheights(root);
                            }
                        }
                        else{
                            if(balanceFactor(previous) < 0){
                                break;
                            }
                            node = previous;
                            continue;
                        }
                    }
                    else{
                        if(balanceFactor(previous) < 0){
                            newPrevious = previous->prev;
                            if(balanceFactor(node) > 0){
                                newRoot = leftRight(previous, node);
                                updateAllheights(root);
                            }
                            else{
                                newRoot = rightRight(previous, node);
                                updateAllheights(root);
                            }
                        }
                        else{
                            if(balanceFactor(previous) > 0){
                                break;
                            }
                            else{
                                node = previous;
                                continue;
                            }
                        }
                    }
                    newRoot->prev = newPrevious;
                    if(newPrevious != nullptr){
                        if(previous == newPrevious->left){
                            newPrevious->left = newRoot;
                        }
                        else{
                            newPrevious->right = newRoot;
                        }
                    }
                    else{
                        root = newRoot;
                    }
                    break;
                }
            }
            int updateAllheights(Node * node){
                if(node == nullptr){
                    return 0;
                }
                else{
                    node->height = 1 + std::max(updateAllheights(node->left), updateAllheights(node->right));
                    return node->height;
                }
            }

            /*void updateheights(Node * current){
                while(current != root){
                            if(current->right != nullptr && current->left != nullptr){
                                if(current->left->height > current->right->height){
                                    current->height = 1 + current->left->height;
                                }
                                else{
                                    current->height = 1 + current->right->height;
                                }
                            }
                            else if(current->right == nullptr && current->left != nullptr){
                                current->height = 1 + current->left->height;
                            }
                            else if(current->right != nullptr && current->left == nullptr){
                                current->height = 1 + current->right->height;
                            }
                            else{
                                current->height = 1;
                            }
                            current = current->prev;
                        }
                        //updating root height
                        if(current->right != nullptr && current->left != nullptr){
                                if(current->left->height > current->right->height){
                                    current->height = 1 + current->left->height;
                                }
                                else{
                                    current->height = 1 + current->right->height;
                                }
                            }
                            else if(current->right == nullptr && current->left != nullptr){
                                current->height = 1 + current->left->height;
                            }
                            else if(current->right != nullptr && current->left == nullptr){
                                current->height = 1 + current->right->height;
                            }
                            else{
                                current->height = 1;
                            }
            }*/
            std::pair<Iterator, bool> insert(const ValueType & value){
                bool inserted = false;
                Node * insert = new Node(value);
                Iterator ret(ret);
                ret.current = insert;
                if(root == nullptr){
                    count = 1;
                    root = insert;
                    inserted = true;
                }
                else{
                    Node * current = root;
                    while(!(current->data.first == value.first) && inserted == false){
                        if(value.first < current->data.first && current->left == nullptr){
                            insert->prev = current;
                            current->left = insert;
                            count ++;
                            inserted = true;
                        }
                        else if(value.first < current->data.first){
                            current = current->left;
                        }
                        else if(current->right == nullptr){
                            insert->prev = current;
                            current->right = insert;
                            count ++;
                            inserted = true;
                        }
                        else{
                            current = current->right;
                        }
                    }
                    if(current->data.first == value.first){
                        ret.current = current;
                        delete insert;
                    }
                    if(inserted){
                        updateAllheights(current);
                        balanceTree(insert);
                        updateAllheights(root);
                    }
                }
                return std::pair<Iterator, bool>(ret, inserted);
            }
            Mapped_T &at(const Key_T & key){
                bool found = false;
                Node * current = root;
                if(root != nullptr){
                    while((current != nullptr) && (found == false)){
                        if(key == current->data.first){
                            found = true;
                        }
                        else if(key < current->data.first){
                            current = current->left;
                        }
                        else{
                            current = current->right;
                        }

                    }
                }
                if(found){
                    return (current->data.second);
                }
                else{
                    throw std::out_of_range("No such Key in map");
                }
            }
            const Mapped_T &at(const Key_T & key) const{
                bool found = false;
                Node * current = root;
                if(root != nullptr){
                    while((current != nullptr) && (found == false)){
                        if(key == current->data.first){
                            found = true;
                        }
                        else if(key < current->data.first){
                            current = current->left;
                        }
                        else{
                            current = current->right;
                        }

                    }
                }
                if(found){
                    return (current->data.second);
                }
                else{
                    throw std::out_of_range("No such Key in map");
                }
            }
            Mapped_T &operator[](const Key_T & key){
                bool found = false;
                Node * current = root;
                if(root != nullptr){
                    while((current != nullptr) && (found == false)){
                        if(key == current->data.first){
                            found = true;
                        }
                        else if(key < current->data.first){
                            current = current->left;
                        }
                        else{
                            current = current->right;
                        }

                    }
                }
                if(found){
                    return (current->data.second);
                }
                Mapped_T mapped = 0;
                std::pair<const Key_T, Mapped_T> input(key, mapped);
                std::pair<Iterator, bool> retPair = insert(input);
                return retPair.first.current->data.second;
            }
            Iterator find(const Key_T & key){
                bool found = false;
                if(root != nullptr){
                    Node * current = root;
                    while((current != nullptr) && (found == false)){
                        if(key == current->data.first){
                            found = true;
                        }
                        else if(key < current->data.first){
                            current = current->left;
                        }
                        else{
                            current = current->right;
                        }

                    }
                    if(found){
                        Iterator ret(ret);
                        ret.current = current;
                        return ret;
                    }
                    return end();
                }
                return end();
            }
            ConstIterator find(const Key_T & key) const{
                bool found = false;
                if(root != nullptr){
                    Node * current = root;
                    while((current != nullptr) && (found == false)){
                        if(key == current->data.first){
                            found = true;
                        }
                        else if(key < current->data.first){
                            current = current->left;
                        }
                        else{
                            current = current->right;
                        }

                    }
                    if(found){
                        ConstIterator ret(ret);
                        ret.current = current;
                        return ret;
                    }
                    return end();
                }
                return end();
            }
            Node * maxofMins(Node * node){
                Node * ret = node->left;
                while(ret->right != nullptr){
                    ret = ret->right;
                }
                return ret;
            }
            void erase(const Key_T & key){
                bool test = false;
                Iterator itr = find(key);
                if(itr.current != nullptr){
                    Node * node = itr.current;
                    //node to be erased is root
                    if(node == root){
                        //root has no children
                        if(node->left == nullptr && node->right == nullptr){
                            root = nullptr;
                        }
                        //root has only left child
                        else if(node->left != nullptr && node->right == nullptr){
                            root = node->left;
                            node->left->prev = nullptr;
                        }
                        //root has only right child
                        else if(node->left == nullptr && node->right != nullptr){
                            root = node->right;
                            node->right->prev = nullptr;
                        }
                        //root has both children
                        else{
                            Node * temp = maxofMins(node);
                            //new root (temp) has a left child (no possible right child)
                            if(temp->left != nullptr){
                                //temp is a left child
                                if(temp->prev == nullptr){
                                    node = temp;
                                }
                                else if(temp->prev->left == temp && temp->prev != node){
                                    temp->prev->left = temp->left;
                                    temp->left->prev = temp->prev;
                                    temp->left = node->left;
                                    temp->right = node->right;
                                    temp->height = node->height;
                                    node->right->prev = temp;
                                    node->right->prev = temp;
                                    if(node->left != nullptr && node->left != temp){
                                        node->left->prev = temp;
                                    }
                                    root = temp;
                                    temp->prev = nullptr;
                                    updateAllheights(root);
                                    balanceTree(temp);
                                }
                                else if(temp->prev->left == temp){
                                    root = temp;
                                    temp->right = node->right;
                                    node->right->prev = temp;
                                    temp->prev = nullptr;
                                    updateAllheights(root);
                                    balanceTree(temp);
                                }
                                //temp is a right child
                                else{
                                    temp->prev->right = temp->left;
                                    temp->left->prev = temp->prev;
                                    temp->left = node->left;
                                    temp->right = node->right;
                                    node->left->prev = temp;
                                    node->right->prev = temp;
                                    root = temp;
                                    temp->prev = nullptr;
                                    updateAllheights(root);
                                    balanceTree(temp);
                                }
                            }
                            //temp has no children
                            else{
                                //temp is a left child
                                if(temp->prev == nullptr){
                                    node = temp;
                                }
                                else if(temp->prev->left == temp && temp->prev != root){
                                    temp->prev->left = nullptr;
                                    temp->left = node->left;
                                    temp->right = node->right;
                                    node->right->prev = temp;
                                    if(node->left != nullptr && node->left != temp){
                                        node->left->prev = temp;
                                    }
                                    root = temp;
                                    temp->prev = nullptr;
                                    updateAllheights(root);
                                    balanceTree(temp);
                                }
                                else if(temp->prev->left == temp){
                                    root = temp;
                                    temp->right = node->right;
                                    node->right->prev = temp;
                                    temp->prev = nullptr;
                                    updateAllheights(root);
                                    balanceTree(temp);
                                }
                                //temp is a right child
                                else if(temp->prev->right == temp && temp->prev != root){
                                    temp->prev->right = nullptr;
                                    temp->left = node->left;
                                    temp->right = node->right;
                                    node->left->prev = temp;
                                    node->right->prev = temp;
                                    root = temp;
                                    temp->prev = nullptr;
                                    updateAllheights(root);
                                    balanceTree(temp);
                                }
                                else{
                                    root = temp;
                                    temp->left = node->left;
                                    node->left->prev = temp;
                                    temp->prev = nullptr;
                                    updateAllheights(root);
                                    balanceTree(temp);
                                }
                            }
                        }
                    }
                    //node is not the root
                    //node has no children
                    else if(node->left == nullptr && node->right == nullptr){
                        //node is left child
                        if(node->prev->left == node){
                            node->prev->left = nullptr;
                            updateAllheights(root);
                            balanceTree(root);
                        }
                        //node is right child
                        else{
                            node->prev->right = nullptr;
                            updateAllheights(root);
                            balanceTree(root);
                        }
                    }
                    //node has only right child
                    else if(node->left == nullptr && node->right != nullptr){
                        //node is left child
                        if(node->prev->left == node){
                            node->prev->left = node->right;
                            node->right->prev = node->prev;
                            updateAllheights(root);
                            balanceTree(root);
                        }
                        //node is right child
                        else{
                            node->prev->right = node->right;
                            node->right->prev = node->prev;
                            updateAllheights(root);
                            balanceTree(root);
                        }
                    }
                    //node has only left child
                    else if(node->left != nullptr && node->right == nullptr){
                        //node is left child
                        if(node->prev->left == node){
                            node->prev->left = node->left;
                            node->left->prev = node->prev;
                            updateAllheights(root);
                            balanceTree(root);
                        }
                        //node is right child
                        else{
                            node->prev->right = node->left;
                            node->left->prev = node->prev;
                            updateAllheights(root);
                            balanceTree(root);
                        }
                    }
                    //node has two children
                    else{
                        Node * temp = maxofMins(node);
                        //new node has no children
                        if(temp->left == nullptr){
                            //temp is left child
                            if(temp->prev->left == temp){
                                temp->prev->left = nullptr;
                                temp->left = node->left;
                                temp->right = node->right;
                                temp->height = node->height;
                                node->right->prev = temp;
                                if(node->left != nullptr && node->left != temp){
                                    node->left->prev = temp;
                                }
                                if(node->prev->left == node){
                                    node->prev->left = temp;
                                }
                                else{
                                    node->prev->right = temp;
                                }
                                temp->prev = node->prev;
                                updateAllheights(root);
                                balanceTree(root);
                            }
                            //temp is right child
                            else{
                                temp->prev->right = nullptr;
                                temp->left = node->left;
                                temp->right = node->right;
                                node->right->prev = temp;
                                node->left->prev = temp;
                                if(node->prev->left == node){
                                    node->prev->left = temp;
                                }
                                else{
                                    node->prev->right = temp;
                                }
                                temp->height = node->height;
                                temp->prev = node->prev;
                                updateAllheights(root);
                                balanceTree(root);
                            }
                        }
                        //new node has left child
                        else{
                            //temp is left child
                            if(temp->prev->left == temp){
                                temp->prev->left = temp->left;
                                temp->left->prev = temp->prev;
                                temp->left = node->left;
                                temp->right = node->right;
                                node->right->prev = temp;
                                temp->prev = node->prev;
                                if(node->left != nullptr && node->left != temp){
                                    node->left->prev = temp;
                                }
                                if(node->prev->left == node){
                                    node->prev->left = temp;
                                }
                                else{
                                    node->prev->right = temp;
                                }
                                updateAllheights(root);
                                balanceTree(root);
                            }
                            //temp is right child
                            else{
                                temp->prev->right = temp->left;
                                temp->left->prev = temp->prev;
                                temp->left = node->left;
                                temp->right = node->right;
                                node->right->prev = temp;
                                node->left->prev = temp;
                                if(node->prev->left == node){
                                    node->prev->left = temp;
                                }
                                else{
                                    node->prev->right = temp;
                                }
                                temp->height = node->height;
                                temp->prev = node->prev;
                                updateAllheights(root);
                                balanceTree(root);
                            }
                        }
                    }
                    count --;
                    delete node;
                    test = true;
                }
                if(test){
                    return;
                }
                else{
                    throw std::out_of_range("No such key in map");
                }
            }
             void erase(Iterator pos){
                bool test = false;
                if(pos.current != nullptr){
                    Node * node = pos.current;
                    erase(node->data.first);
                    test = true;
                }
                if(test){
                    return;
                }
                else{
                    throw std::out_of_range("No such key in map");
                }
             }
             void clear(){
                while(count > 0){
                    erase(root->data.first);
                }
                root = nullptr;
             }
             Map(std::initializer_list<std::pair<const Key_T, Mapped_T>> pairs){
                count = 0;
                root = nullptr;
                for(ValueType insertPair : pairs){
                    insert(insertPair);
                }
            }
            template <typename IT_T> void insert(IT_T range_beg, IT_T range_end){
                count = 0;
                root = nullptr;
                for(auto i = range_beg; i != range_end; i++){
                    insert(*i);
                }
            }
             ~Map(){
                clear();
                delete root;
            }
            ValueType getRoot(){
                return root->data;
            }

            Node * getRoot2(){
                return root;
            }
    };   
}     

#endif
