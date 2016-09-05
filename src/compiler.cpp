#include "compiler.h"
#include <sys/stat.h>
#include <cxxabi.h>

std::string get_call_args(std::size_t num_args) {
    std::string call_args;
    for (int i = 0; i < num_args; i++) {
        if (i > 0) {
            call_args = call_args + ", ";
        }
        call_args = call_args + (char)(((int)'a') + i);
    }
    return call_args;
}

bool file_exists (const std::string& fname) {
    struct stat buffer;
    return (stat (fname.c_str(), &buffer) == 0);
}

std::string get_class_name(const char* name) {
    int status;
    char * demangled = abi::__cxa_demangle(
        name,
        0,
        0,
        &status
    );
    return std::string(demangled);
}

ModulePointer::ModulePointer(const std::string& libname) : module_(NULL), libname_(libname) {
    module_ = dlopen(libname_.c_str(), RTLD_LAZY);
    if(!module_) {
        std::cerr << "error loading library:\n" << dlerror() << std::endl;
        exit(EXIT_FAILURE);
    }
}

ModulePointer::~ModulePointer() {
    if (module_) {
        dlclose(module_);
    }
}

Module::Module() : module_ptr_(NULL) {}
Module::Module(const std::string& libname) :
        module_ptr_(std::make_shared<ModulePointer>(libname)) {
}

void* Module::module() {
    return module_ptr_->module_;
}

Compiler::Compiler(const std::vector<Headerfile>& headerfiles, const std::string& include_dir, const std::string& outpath)
        : headerfiles_(headerfiles), include_dir_(include_dir), outpath_(outpath) {
    copy_headers();
}

void Compiler::copy_headers() const {
    for (auto& header : headerfiles_) {
        system(
            make_message("cp ", header.path_, " ", outpath_).c_str()
        );
    }
}

std::string Compiler::header_file_includes() const {
    std::stringstream ss;
    for (auto& header : headerfiles_) {
        ss << "#include \"" << header.name_ << "\"\n";
    }
    return ss.str();
}

bool Compiler::compile_code(const std::string& source,
                  const std::string& dest,
                  const std::string& logfile) {
    Timer t1("nvcc " + source);
    std::string cmd = make_message(
        "nvcc -std=c++11 ", source,
        " -o ", dest,
        " -I", include_dir_,
        // dynamic lookup: -undefined dynamic_lookup on clang
        // and -Wl,--unresolved-symbols=ignore-in-object-files for gcc
        " --compiler-options=\"-undefined dynamic_lookup\" "
        " -O2 -shared &> ",logfile
    );
    int ret = system(cmd.c_str());
    return WEXITSTATUS(ret) == EXIT_SUCCESS;
}
