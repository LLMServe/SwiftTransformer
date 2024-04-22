#pragma once

#include <cassert>

/*
assert_whenever: assertion which ignore whether NDEBUG is set

In C++, assert() is evaluated only when NDEBUG is not set. This is
inconvenient when we want to check the assertion even in release mode.
This macro is a workaround for this problem.
*/

extern "C" {
// Copied from assert.h
extern void __assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
     __THROW __attribute__ ((__noreturn__));

#define __ST_ASSERT_FUNCTION	__extension__ __PRETTY_FUNCTION__
#  define assert_whenever(expr)							\
     (static_cast <bool> (expr)						\
      ? void (0)							\
      : __assert_fail (#expr, __FILE__, __LINE__, __ST_ASSERT_FUNCTION))
}
