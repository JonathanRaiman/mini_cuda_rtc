#ifndef RTC_COMPILER_H
#define RTC_COMPILER_H

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cstdlib> // EXIT_FAILURE, etc
#include <tuple>
#include <dlfcn.h>      // dynamic library loading, dlopen() etc
#include "timer.h"
#include "make_message.h"

std::string get_call_args(std::size_t num_args);
bool file_exists(const std::string& fname);
std::string get_class_name(const char* name);

template<typename... Args, typename std::enable_if<sizeof... (Args) == 0, int>::type = 0>
void get_function_arguments(int i, std::string* call_ptr) {}

template<typename Arg, typename... Args>
void get_function_arguments(int i, std::string* call_ptr) {
    std::string& call = *call_ptr;
    if (i > 0) {
        call = call + ", ";
    }
    call = call + get_class_name(typeid(Arg).name()) + " " + (char)(((int)'a') + i);
    get_function_arguments<Args...>(i+1, call_ptr);
}

template<typename... Args>
std::string get_function_arguments() {
    std::string s;
    get_function_arguments<Args...>(0, &s);
    return s;
}

struct ModulePointer {
    void* module_;
    std::string libname_;
    ModulePointer(const std::string& libname);
    ~ModulePointer();
};

struct Module {
    std::shared_ptr<ModulePointer> module_ptr_;

    Module();
    Module(const std::string& libname);
    void* module();

    template<typename T>
    T get_symbol(const std::string& name) {
        Timer t1("get_symbol");
        void* symbol = dlsym(module(), name.c_str());
        const char* dlsym_error = dlerror();
        if (dlsym_error != NULL) {
            std::cerr << "error loading symbol:\n" << dlsym_error << std::endl;
            exit(EXIT_FAILURE);
        }
        return reinterpret_cast<T>(symbol);
    }
};

namespace std {
    template<typename... TTypes>
    class hash<std::tuple<TTypes...>> {
        private:
            typedef std::tuple<TTypes...> Tuple;

            template<int N>
            size_t operator()(Tuple value) const {
                return 0;
            }

            template<int N, typename THead, typename... TTail>
            size_t operator()(Tuple value) const {
                constexpr int Index = N - sizeof...(TTail) - 1;
                return hash<THead>()(std::get<Index>(value)) ^ operator()<N, TTail...>(value);
            }

        public:
            size_t operator()(Tuple value) const {
                return operator()<sizeof...(TTypes), TTypes...>(value);
            }
    };
}

struct Headerfile {
    std::string path_;
    std::string name_;
    Headerfile(const std::string& path, const std::string& name) :
        path_(path), name_(name) {}
};

struct Compiler {
    std::vector<Headerfile> headerfiles_;
    std::string outpath_;
    std::string include_dir_;
    std::unordered_map<std::tuple<std::size_t, std::size_t>, Module> modules;

    Compiler(const std::vector<Headerfile>& headerfiles,
             const std::string& include_dir,
             const std::string& outpath);

    void copy_headers() const;
    std::string header_file_includes() const;

    template<typename... Args>
    void write_code(const std::string& fname,
                    const std::string& code,
                    const std::string& funcname) {
        Timer t1("write_code " + fname);

        std::ofstream out(fname.c_str(), std::ofstream::out);
        if (out.bad()) {
            std::cout << "cannot open " << fname << std::endl;
            exit(EXIT_FAILURE);
        }
        // add header to code (and extern c to avoid name mangling)
        std::string newcode = make_message(
            header_file_includes(),
            code, "\n", "extern \"C\" void maker (",
            get_function_arguments<Args...>(),
            "){\n", funcname, '(', get_call_args(sizeof...(Args)), ");}"
        );
        out << newcode;
        out.flush();
        out.close();
    }

    bool compile_code(const std::string& source,
                      const std::string& dest,
                      const std::string& logfile);

    template<typename... Args>
    void create_module(const std::string& save_name,
                       const std::string& code,
                       const std::string& funcname,
                       bool force_recompilation,
                       const std::tuple<std::size_t, std::size_t>& module_key) {
        Timer t1("nvcc+load " + save_name);

        std::string libfile = save_name + ".so";
        bool module_never_compiled = !file_exists(libfile);
        if (force_recompilation || module_never_compiled) {
            std::cout << "Compiling..." << std::endl;
            std::string cppfile = save_name + ".cu";
            std::string logfile = save_name + ".log";

            write_code<Args...>(cppfile, code, funcname);
            bool success = compile_code(
                cppfile,
                libfile,
                logfile
            );

            if (!success) {
                std::cout << "Compilation failed, see " << logfile << std::endl;
                exit(EXIT_FAILURE);
            }
        } else {
            std::cout << "Module previously compiled: reusing." << std::endl;
        }
        Module module(libfile);
        modules[module_key] = module;
    }

    // compile code, instantiate class and return pointer to base class
    // https://www.linuxjournal.com/article/3687
    // http://www.tldp.org/HOWTO/C++-dlopen/thesolution.html
    // https://stackoverflow.com/questions/11016078/
    // https://stackoverflow.com/questions/10564670/
    template<typename... Args>
    std::function<void(Args...)> compile(
            const std::string& code,
            std::string funcname,
            bool force_recompilation) {

        Timer code_hash_timer("code_hash_timer");

        std::size_t code_hash = std::hash<std::string>()(code);
        std::string func_args = get_function_arguments<Args...>();
        std::size_t arg_hash  = std::hash<std::string>()(func_args);
        std::tuple<std::size_t, std::size_t> module_key(code_hash, arg_hash);

        code_hash_timer.stop();

        bool module_never_loaded = modules.find(module_key) == modules.end();

        if (force_recompilation || module_never_loaded) {
            create_module<Args...>(
                make_message(outpath_, "/", code_hash, arg_hash),
                code, funcname,
                force_recompilation,
                module_key
            );
        } else {
            std::cout << "Module previously loaded: reusing." << std::endl;
        }
        std::function<void(Args...)> method = modules[module_key].get_symbol<void(*)(Args...)>("maker");
        return method;
    }
};

#endif
