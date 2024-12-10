
#ifndef VECMEM_CORE_EXPORT_H
#define VECMEM_CORE_EXPORT_H

#ifdef VECMEM_CORE_STATIC_DEFINE
#  define VECMEM_CORE_EXPORT
#  define VECMEM_CORE_NO_EXPORT
#else
#  ifndef VECMEM_CORE_EXPORT
#    ifdef vecmem_core_EXPORTS
        /* We are building this library */
#      define VECMEM_CORE_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define VECMEM_CORE_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef VECMEM_CORE_NO_EXPORT
#    define VECMEM_CORE_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef VECMEM_CORE_DEPRECATED
#  define VECMEM_CORE_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef VECMEM_CORE_DEPRECATED_EXPORT
#  define VECMEM_CORE_DEPRECATED_EXPORT VECMEM_CORE_EXPORT VECMEM_CORE_DEPRECATED
#endif

#ifndef VECMEM_CORE_DEPRECATED_NO_EXPORT
#  define VECMEM_CORE_DEPRECATED_NO_EXPORT VECMEM_CORE_NO_EXPORT VECMEM_CORE_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef VECMEM_CORE_NO_DEPRECATED
#    define VECMEM_CORE_NO_DEPRECATED
#  endif
#endif

#endif /* VECMEM_CORE_EXPORT_H */
