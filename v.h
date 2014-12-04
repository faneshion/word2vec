#pragma once
#include<vector>
#include<assert.h>

typedef std::vector<float> Vector;

namespace v{

    template <class Vector1,class Vector2> inline float dot(const Vector1& x,const Vector2& y){
        int m = x.size();
        const float *xd = x.data(), *yd = y.data();
        float sum = 0.0;
        while( --m >= 0 ) sum += (*xd++ ) * (*yd++);
        return sum;
    }

    // x = x + g * y
    inline void saxpy(Vector &x,float g, const Vector& y){
        assert( x.size() == y.size());
        int m = x.size();
        float *xd = x.data();
        const float *yd = y.data();
        while( --m >= 0){
            *xd += g * (*yd); ++xd; ++yd;
        }
    }

    // x = a * x + g * y
    inline void saxpy(float a,Vector& x,float g, const Vector &y){
        assert(x.size() == y.size());
        int m = x.size();
        float *xd = x.data();
        const float *yd = y.data();
        while( --m >=0 ){ (*xd) =a * (*xd) + g * (*yd); ++xd;++yd;}
    }
    inline void unit(Vector &x){
        float len = ::sqrt(dot(x,x));
        if(len == 0 ) return;
        int m = x.size();
        float *xd = x.data();
        while(--m >= 0 ) (*xd++) /= len;
    }
}

