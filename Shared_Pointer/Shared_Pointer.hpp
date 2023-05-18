#ifndef SHAREDPTR_HPP
#define SHAREDPTR_HPP
#include <stdlib.h>
#include <utility>
#include <stdexcept>
#include <stdio.h>
#include <ostream>
#include <iostream>
#include <mutex>
#include <atomic>

namespace cs440{
    //base class to delete the original pointer type
    class Base{
        public:
            virtual void deletePtr(){};
            virtual ~Base(){};
    };
    //templated class inherited from base to hold the actual pointer
    template <typename T> class originalPtr : public Base{
        private:
            T * ptr;
        public:
            originalPtr(){
                ptr = nullptr;
            }
            originalPtr(T * p){
                ptr = p;
            }
            //make virtual to call from base
            void deletePtr() override{
                delete ptr;
            }
    };
    //Shared reference object that holds all shared attributes of the shared ptr
    class referenceCounter{
        private:
            //automatically is atomic no more need for locks
            //around increments and decrements
            std::atomic<int> count{0};
            Base * basePtr;
        public:
            referenceCounter(){
                //counterLock.lock();
                count = 0;
                basePtr = nullptr;
                //counterLock.unlock();
            }
            template <typename T> referenceCounter(T * p){
                count = 1;
                basePtr = new originalPtr<T>(p);
            }
            void increaseCount(){
                //counterLock.lock();
                count++;
                //counterLock.unlock();
            }
            void decreaseCount(){
                //counterLock.lock();
                count--;
                //counterLock.unlock();
            }
            int getReferenceCount(){
                return count;
            }
            ~referenceCounter(){
                if(basePtr != nullptr){
                    basePtr->deletePtr();
                    delete basePtr;
                }
            }
    };
    //Shared pointer class
    template <typename T> class SharedPtr{
        private:
            referenceCounter * counter;
            T * pointer;
        public:
            SharedPtr(){
                pointer = nullptr;
                counter = nullptr;
            }
            template <typename U> explicit SharedPtr(U * object){
                counter = new referenceCounter(object);
                pointer = object;
            }
            SharedPtr(const SharedPtr & p){
                if(p.getCounter() != nullptr){
                    p.getCounter()->increaseCount();
                }
                counter = p.getCounter();
                pointer = p.get();
            }
            template <typename U> SharedPtr(const SharedPtr<U> & p){
                if(p.getCounter() != nullptr){
                    p.getCounter()->increaseCount();
                }
                counter = p.getCounter();
                pointer = p.get();
            }
            SharedPtr(SharedPtr &&p){
                pointer = p.get();
                counter = p.getCounter();
                p.pointerReset();
                p.counterReset();
            }
            template <typename U> SharedPtr(SharedPtr<U> &&p){
                pointer = p.get();
                counter = p.getCounter();
                p.pointerReset();
                p.counterReset();
            }
            SharedPtr &operator=(const SharedPtr & p){
                if(counter != nullptr){
                    counter->decreaseCount();
                    if(counter->getReferenceCount() == 0){
                        delete counter;
                        counter = nullptr;
                        pointer = nullptr;
                    }
                }
                if(p != nullptr){
                    counter = p.getCounter();
                    if(counter != nullptr){
                        counter->increaseCount();
                    }
                    pointer = p.get();
                }
                return *this;
            }
            template <typename U> SharedPtr<T> &operator=(const SharedPtr<U> & p){
                if(counter != nullptr){
                    counter->decreaseCount();
                    if(counter->getReferenceCount() == 0){
                        delete counter;
                        counter = nullptr;
                        pointer = nullptr;
                    }
                }
                if(p != nullptr){
                    counter = p.getCounter();
                    if(counter != nullptr){
                        counter->increaseCount();
                    }
                    pointer = p.get();
                }
                return *this;
            }
            SharedPtr &operator=(SharedPtr &&p){
                if(counter != nullptr){
                    counter->decreaseCount();
                    if(counter->getReferenceCount() == 0){
                        delete counter;
                        counter = nullptr;
                        pointer = nullptr;
                    }
                }
                counter = p.getCounter();
                pointer = p.get();
                p.counterReset();
                p.pointerReset();
                return *this;
            }
            template <typename U> SharedPtr &operator=(SharedPtr<U> &&p){
                if(counter != nullptr){
                    counter->decreaseCount();
                    if(counter->getReferenceCount() == 0){
                        delete counter;
                        counter = nullptr;
                        pointer = nullptr;
                    }
                }
                counter = p.getCounter();
                pointer = p.get();
                p.counterReset();
                p.pointerReset();
                return *this;
            }
            ~SharedPtr(){
                if(counter != nullptr){
                    counter->decreaseCount();
                    if(counter->getReferenceCount() == 0){
                        delete counter;
                        counter = nullptr;
                        pointer = nullptr;
                    }
                }
            }

            void reset(){
                if(counter != nullptr){
                    counter->decreaseCount();
                    if(counter->getReferenceCount() == 0){
                        delete counter;
                        counter = nullptr;
                        pointer = nullptr;
                    }
                }
                pointer = nullptr;
                counter = nullptr;
            }
            template <typename U> void reset(U * p){
                if(counter != nullptr){
                    counter->decreaseCount();
                    if(counter->getReferenceCount() == 0){
                        delete counter;
                        counter = nullptr;
                        pointer = nullptr;
                    }
                }
                pointer = p;
                counter = new referenceCounter(p);

            }
            T *get() const{
                return pointer;
            }
            T &operator*() const{
                return *pointer;
            }
            T *operator->() const{
                return pointer;
            }
            explicit operator bool() const{
                return pointer != nullptr;
            }
            int getReferenceCount(){
                if(counter != nullptr){
                    return counter->getReferenceCount();
                }
                return -1;
            }
            referenceCounter * getCounter() const{
                return counter;
            }
            void setPointer(T * p){
                pointer = p;
            }
            void setCounter(referenceCounter * c){
                counter = c;
            }
            void pointerReset(){
                pointer = nullptr;
            }
            void counterReset(){
                counter = nullptr;
            }
    };      
    template <typename T1, typename T2> bool operator==(const SharedPtr<T1> & p1, const SharedPtr<T2> & p2){
        if((p1.get() == nullptr && p2.get() == nullptr) || (p1.get() == p2.get())){
            return true;
        }
        return false;
    }
    template <typename T> bool operator==(const SharedPtr<T> & p, std::nullptr_t){
        if(p.get() == nullptr){
            return true;
        }
        return false;
    }
    template <typename T> bool operator==(std::nullptr_t, const SharedPtr<T> & p){
        if(p.get() == nullptr){
            return true;
        }
        return false;
    }
    template <typename T1, typename T2> bool operator!=(const SharedPtr<T1>& p1, const SharedPtr<T2> & p2){
        return !(p1 == p2);
    }
    template <typename T> bool operator!=(const SharedPtr<T> & p, std::nullptr_t){
        return !(p == nullptr);
    }
    template <typename T> bool operator!=(std::nullptr_t, const SharedPtr<T> & p){
        return !(nullptr == p);
    }
    template <typename T, typename U> SharedPtr<T> static_pointer_cast(const SharedPtr<U> &sp){
        SharedPtr<T> ret;
        ret.setPointer(static_cast<T*>(sp.get()));
        ret.setCounter(sp.getCounter());
        return ret;
    }
    template <typename T, typename U> SharedPtr<T> dynamic_pointer_cast(const SharedPtr<U> &sp){
        SharedPtr<T> ret;
        ret.setPointer(dynamic_cast<T*>(sp.get()));
        ret.setCounter(sp.getCounter());
        return ret;
    }
}

#endif
