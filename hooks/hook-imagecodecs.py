from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

binaries = collect_dynamic_libs('imagecodecs')
hiddenimports = collect_submodules('imagecodecs')
