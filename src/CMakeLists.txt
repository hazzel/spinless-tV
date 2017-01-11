# Start configuration
cmake_minimum_required(VERSION 2.8)
project(mqmc)
MESSAGE("MCLL_SYSTEM_INFO set to $ENV{MCLL_SYSTEM_INFO}")
set(mode "mpi")

enable_language(CXX)

# Add load leveller source directory
set(MCLL "$ENV{HOME}/mc/load_leveller/trunk")
MESSAGE("load_leveller source directory set to ${MCLL}")

# Linking and include info
include_directories(${CMAKE_SOURCE_DIR})
include_directories($ENV{HOME}/libs/eigen $ENV{HOME}/eigen)
include_directories(${MCLL})

if(mode STREQUAL "mpi")
	# Set up MPI
	MESSAGE("Using MPI implementation.")
	if("$ENV{MCLL_SYSTEM_INFO}" STREQUAL "juqueen")
		#set(MPI_CXX_COMPILER /bgsys/drivers/ppcfloor/comm/xl.ndebug/bin/mpixlcxx)
		set(MPI_CXX_COMPILER /bgsys/local/gcc/4.8.3/bin/mpig++)
		#set(MPI_CXX_COMPILER /bgsys/drivers/ppcfloor/comm/xl.ndebug/bin/mpic++)
	endif()
	FIND_PACKAGE(MPI REQUIRED)
	if (MPI_CXX_FOUND)
		set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_CXX_COMPILE_FLAGS}")
		set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_CXX_LINK_FLAGS}")
		MESSAGE("Using mpi wrapper ${MPI_CXX_COMPILER}")

		include_directories(${MPI_CXX_INCLUDE_PATH})
	else()
		MESSAGE("MPI not found.")
	endif()

	file(GLOB LL_SRC ${MCLL}/dump.cpp
						${MCLL}/parser.cpp
						${MCLL}/measurements.cpp
						${MCLL}/evalable.cpp
						${MCLL}/observable.cpp
						${MCLL}/random.cpp
						${MCLL}/main.cpp
						${MCLL}/runner.cpp)
elseif(mode STREQUAL "single")
	MESSAGE("Using single core implementation.")
	file(GLOB LL_SRC ${MCLL}/dump.cpp
						${MCLL}/parser.cpp
						${MCLL}/measurements.cpp
						${MCLL}/evalable.cpp
						${MCLL}/observable.cpp
						${MCLL}/random.cpp
						${MCLL}/main.cpp
						${MCLL}/runner_single.cpp)
	set(DEFINES "${DEFINES} -DMCL_SINGLE")
endif()

FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )

# Compiler flags
set(DEFINES "${DEFINES} -DMCL_DUMP_BUFFER=0 -DMCL_MEASUREMENTS_APPEND=0 -DMCL_MCL_RNG_MT -ftree-vectorizer-verbose=0")
if("$ENV{MCLL_SYSTEM_INFO}" STREQUAL "rwthcluster")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} $ENV{FLAGS_FAST} -DMKL_DIRECT_CALL -std=c++11 ${DEFINES} -DEIGEN_USE_MKL_ALL")
	set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} $ENV{FLAGS_MKL_INCLUDE}")
	set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} $ENV{FLAGS_MKL_LINKER}")
elseif("$ENV{MCLL_SYSTEM_INFO}" STREQUAL "juqueen")
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -qarch=qp -qtune=qp -qstrict -qlanglvl=extended0x ${DEFINES}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native -std=c++11 ${DEFINES}")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -std=c++11 -pipe -g ${DEFINES}")
	find_library(PROFILER NAMES profiler PATHS $ENV{HOME}/gperftools-2.5/install/lib)
	include_directories($ENV{HOME}/gperftools-2.5/install/include)
	message("Using profiler: ${PROFILER}")
endif()

# Create executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
add_executable(mqmc ${LL_SRC} mc.cpp)
if (PROFILER)
	target_link_libraries(mqmc ${MPI_LIBRARIES} ${Boost_LIBRARIES} ${PROFILER})
else()
	target_link_libraries(mqmc ${MPI_LIBRARIES} ${Boost_LIBRARIES})
endif()
