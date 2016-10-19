# 增加opencv的依赖
FIND_PACKAGE( OpenCV REQUIRED )

# 添加头文件和库文件
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS})


# build lib

ADD_LIBRARY( tld_utils tld_utils.cpp )
ADD_LIBRARY( LKTracker LKTracker.cpp )
ADD_LIBRARY( FerNN FerNNClassifier.cpp )
ADD_LIBRARY( tld TLD.cpp )

# build exe
ADD_EXECUTABLE( run_tld  run_tld.cpp )
TARGET_LINK_LIBRARIES( run_tld 
    tld
    LKTracker
    FerNN
    tld_utils
    ${OpenCV_LIBS} 
    )
