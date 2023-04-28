add_mlir_dialect_library(MLIRFlow
        FlowDialect.cpp
        FlowOps.cpp
        Traits.cpp
        FlowTypes.cpp
        LowerToLLVM.cpp
        LowerToAffineLoops.cpp
        ShapeInference.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRFlowOpsIncGen # FlowOps
        CombineIncGen
        ShapeInferenceInterfaceIncGen

        LINK_LIBS PUBLIC

        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        )