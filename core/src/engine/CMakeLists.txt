set(SOURCE_CORE_ENGINE
    ${SOURCE_CORE_ENGINE}
    ${CMAKE_CURRENT_SOURCE_DIR}/Neighbours.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Force_LLG.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Force_GNEB.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Hamiltonian_Isotropic.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Hamiltonian_Anisotropic.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Optimizer_SIB.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Optimizer_SIB2.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Optimizer_Heun.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Optimizer_CG.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Optimizer_QM.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Method.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Method_LLG.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Method_GNEB.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Method_MMF.cpp
    PARENT_SCOPE # needed so the change of ${SOURCE} will persist to the parent scope
)