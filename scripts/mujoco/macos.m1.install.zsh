
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export CC="/opt/homebrew/opt/llvm/bin/clang"
export CXX="/opt/homebrew/opt/llvm/bin/clang++"
export CXX11="/opt/homebrew/opt/llvm/bin/clang++"
export CXX14="/opt/homebrew/opt/llvm/bin/clang++"
export CXX17="/opt/homebrew/opt/llvm/bin/clang++"
export CXX1X="/opt/homebrew/opt/llvm/bin/clang++"
export LDFLAGS="-L/opt/homebrew/opt/llvm/lib"
export CPPFLAGS="-I/opt/homebrew/opt/llvm/include"

pip3 install -U mujoco-py
pip3 install -U mujoco
