# Mac OS

## Prerequisites

The prerequisites listed below are required to be able to configure/build/package/test Slicer.

- XCode command line tools must be installed:
```
xcode-select --install
```
- A CMake version that meets at least the minimum required CMake version [here](https://github.com/Slicer/Slicer/blob/master/CMakeLists.txt#L1)
- Large File Storage for git is required. (`brew install git-lvs`)
- Qt 5: **tested and recommended**.
  - For building Slicer: download and execute [qt-unified-mac-x64-online.dmg](https://download.qt.io/official_releases/online_installers/qt-unified-mac-x64-online.dmg), install Qt 5.15, make sure to select `qtscript` and `qtwebengine` components.
  - For packaging and redistributing Slicer: build Qt using [qt-easy-build](https://github.com/jcfr/qt-easy-build#readme)
- Setting `CMAKE_OSX_DEPLOYMENT_TARGET` CMake variable specifies the minimum macOS version a generated installer may target.  So it should be equal to or less than the version of SDK you are building on. Note that the SDK version is set using `CMAKE_OSX_SYSROOT` CMake variable automatically initialized during CMake configuration.

## Checkout Slicer source files

Notes:
- While it is not enforced, we strongly recommend you to *avoid* the use of *spaces* for both the `source directory` and the `build directory`.
- Due to maximum path length limitations during build the build process, source and build folders must be located in a folder with very short total path length. This is expecially critical on Windows and MacOS. For example, `/sq5` has been confirmed to work on MacOS.

Check out the code using `git`:
- Clone the github repository</p>
```cd MyProjects
git clone git://github.com/Slicer/Slicer.git
```
The `Slicer` directory is automatically created after cloning Slicer.
- Setup the development environment:
```cd Slicer
./Utilities/SetupForDevelopment.sh
```

## Configure and generate Slicer solution files

- Configure using the following commands. By default `CMAKE_BUILD_TYPE` is set to `Debug` (replace `/path/to/QtSDK` with the real path on your machine where QtSDK is located):
```
mkdir Slicer-SuperBuild-Debug
cd Slicer-SuperBuild-Debug
cmake -DCMAKE_BUILD_TYPE:STRING=Debug -DQt5_DIR:PATH=/path/to/Qt5.15.0/5.15.0/gcc_64/lib/cmake/Qt5 ../Slicer
```
- If `using Qt from the system`, do not forget to add the following CMake variable to your configuration command line: `-DSlicer_USE_SYSTEM_QT:BOOL=ON`
- Remarks:
  - Instead of `cmake`, you can use `ccmake` or `cmake-gui` to visually inspect and edit configure options.
  - Using top-level directory name like `Slicer-SuperBuild-Release` or `Slicer-SuperBuild-Debug` is recommended.
  - [Step-by-step debug instuctions](https://www.slicer.org/wiki/Documentation/Nightly/Developers/Tutorials/Debug_Instructions)
  - Additional configuration options to customize the application are described [here](overview.md#Customized_builds).

### General information

Two projects are generated by either `cmake`, `ccmake` or `cmake-gui`. One of them is in the top-level bin directory `Slicer-SuperBuild` and the other one is in the subdirectory `Slicer-build`:
- `Slicer-SuperBuild` manages all the external dependencies of Slicer (VTK, ITK, Python, ...). To build Slicer for the first time, run make (or build the solution file in Visual Studio) in `Slicer-SuperBuild`, which will update and build the external libraries and if successful will then build the subproject Slicer-build.
- `Slicer-SuperBuild/Slicer-build` is the "traditional" build directory of Slicer.  After local changes in Slicer (or after an svn update on the source directory of Slicer), only running make (or building the solution file in Visual Studio) in `Slicer-SuperBuild/Slicer-build` is necessary (the external libraries are considered built and up to date).

*Warning:* An significant amount of disk space is required to compile Slicer in Debug mode (>10GB)

*Warning:* Some firewalls will block the git protocol. See more information and solution [here](../overview.html#firewall-is-blocking-git-protocol).

## Build Slicer

After configuration, start the build process in the `Slicer-SuperBuild` directory

- Start a terminal and type the following (you can replace 4 by the number of processor cores in the computer):
```
cd ~/Projects/Slicer-SuperBuild
make -j4
```

When using the -j option, the build will continue past the source of the first error. If the build fails and you don't see what failed, rebuild without the -j option. Or, to speed up this process build first with the -j and -k options and then run plain make. The -k option will make the build keep going so that any code that can be compiled independent of the error will be completed and the second make will reach the error condition more efficiently.

## Run Slicer

Start a terminal and type the following:
```
Slicer-SuperBuild/Slicer-build/Slicer
```

## Test Slicer

After building, run the tests in the  `Slicer-SuperBuild/Slicer-build` directory.

Start a terminal and type the following (you can replace 4 by the number of processor cores in the computer):
```
cd ~/Projects/Slicer-SuperBuild/Slicer-build
ctest -j4
```

## Package Slicer

Start a terminal and type the following:
```
cd ~/Projects/Slicer-SuperBuild
cd Slicer-build
make package
```

## Common errors

See list of issues common to all operating systems on [Common errors](common_errors) page.

### CMake complains during configuration

CMake may not directly show what's wrong; try to look for log files of the form BUILD/CMakeFiles/*.log (where BUILD is your build directory) to glean further information.

### 'QSslSocket' : is not a class or namespace name

This error message occurs if Slicer is configured to use SSL but Qt is built without SSL support.

Either set Slicer_USE_PYTHONQT_WITH_OPENSSL to OFF when configuring Slicer build in CMake, or build Qt with SSL support.

### macOS: error while configuring PCRE: "cannot run C compiled program"

If the XCode command line tools are not properly set up on macOS, PCRE could fail to build in the Superbuild process with the errors like below:
```
configure: error: in `/Users/fedorov/local/Slicer4-Debug/PCRE-build':
configure: error: cannot run C compiled programs.
```

To install XCode command line tools, use the following command from the terminal:
```
xcode-select --install
```

### macOS: dyld: malformed mach-o: load commands size (...) > 32768

Path of source or build folder is too long. For example building Slicer in */User/somebody/projects/something/dev/slicer/slicer-qt5-rel* may fail with malformed mach-o error, while it succeeds in */sq5* folder. To resolve this error, move source and binary files into a folder with shorter full path and restart the build from scratch (the build tree is not relocatable).
