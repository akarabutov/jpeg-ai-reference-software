# Entropy coding development

## General recommendations

There are several examples of entropy engines and probability models inside existing SW. It is highly recommended to follow the same style, structure and naming conventions used in provided code base.

## Adding new entropy engine

The steps for including new entropy engine into the code are described below. The lib making steps are described in [readme](src/codec/entropy_coding/core_libs/cpp/README.md).

###	class ECInstanceFactory(object)

Add instance of new entropy engine to the factory with some arbitrary non-overlapping index (**111** in example):

    def __init__(self, *args, **kwargs):
            self.modules = {
                'range':          decorator_with_coder_type(1, *args, **kwargs)(ECLibCpp),
                'real':             decorator_with_coder_type(0, *args, **kwargs)(ECLibCpp),
                'int_range':   decorator_with_coder_type(4, *args, **kwargs)(ECLibCpp),
                **'new_ngn'**:   decorator_with_coder_type(111, *args, **kwargs)(ECLibCpp),
                'lh':                 ECLibProxy
            }

###	Make new bitstream IO if needed
If some new storage (like memory) or transmission is required then prepare an analogue of BitOutputStream class for such reasons in [src/codec/entropy_coding/core_libs/cpp/bit_streams]

###	Prepare new coding engine
Create new entropy engine in folder `src/codec/entropy_coding/core_libs/cpp/coding_algos/`**new_ngn**.
Good to follow the way containing base class NewNgn_coder, class for encoder NewNgn_encoder and class for decoder NewNgn_decoder for new engine.

        class NewNgnCoder {
            // Constructs a base class for engine of new entropy coder, which initializes state of the coder.
            explicit NewNgnCoder(parameters);
            virtual ~NewNgnCoder() = 0;
            …
        }

        class NewNgnEncoder final : public NewNgnCoder {
            explicit NewNgnEncoder(parameters);
            …
        }
        class NewNgnDecoder final : public NewNgnCoder {
            explicit NewNgnDecoder(parameters);
            …
        }

###	Add initialization and releasing of new engine
Update initialization and releasing according new engine needs in [src/codec/entropy_coding/core_libs/cpp/coding_module/base_wrapper.cpp]

        EXPORTED void NewEncoder(const int encType, const int debugId, const long long startDbg)
        {	…
            switch (g_coderType) {
                case 0: {
                    pBitOut = new BitOutputStream(*pFileOut);
                    pAcEnc = new ArithmeticEncoder(32, *pBitOut);
                    break;
                }
                case 111: {
                    pBitOutMem = new uint16_t[100000000];
                    pNewNgnEnc = new NewNgnEnc(pBitOutMem);
                    break;
                }
            …

        EXPORTED void ClearEncoder()
        {
            switch (g_coderType) {
                case 0 : { pAcEnc->Finish(); pAcEnc = nullptr; break; }
                case 111 : { pNewNgnEnc->Finish(); pNewNgnEnc = nullptr; break; }
                default: { printf("Unsupported encoder type.\n"); break; }
            }
            …

        EXPORTED void DelEncoder()
        {
            if (pAcEnc) {
                delete pAcEnc;
                pAcEnc = nullptr;
            }
            …
            if (pNewNgnEnc) {
                delete pNewNgnEnc;
                pNewNgnEnc = nullptr;
            }
            …


        EXPORTED void NewDecoder(const int decType, const int debugId, const long long startDbg)
        {
            …
            switch (g_coderType) {
                case 0: {
                    pBitIn = new BitInputStream(*pFileIn, 1);
                    pAcDec = new ArithmeticDecoder(32, *pBitIn);
                    break;
                }
                case 111: {
                    pNewNgnDec = new NewNgnDec( pBitInMem );
                    break;
                }
                default: {
                    printf("Unsupported decoder type\n");
                    break;
                }
            }
            …


        EXPORTED void ClearDecoder()
        {
            …
            switch (g_coderType) {
                case 0: { pAcDec = nullptr; break; }
                case 111: { pNewNgnDec = nullptr; break; }
                default: { printf("Unsupported decoder type.\n"); break; }
            }
            …


        EXPORTED void DelDecoder()
        {
            if (pAcDec) {
                delete pAcDec;
                pAcDec = nullptr;
            }
            …
            if (pNewNgnDec) {
                delete pNewNgnDec;
                pNewNgnDec = nullptr;
            }
            …


        EXPORTED void GetState(long *pLow, long *pRange, long *pCode)
        {
            if (pAcEnc != nullptr) {
                std::cout << "AcEnc is be implemented!" <<
                (long)pAcEnc->GetLow() <<
                (long)pAcEnc->GetRange() <<
                (long)pAcEnc->GetCode() << std::endl;
                …
            } else if (pNewNgnEnc != nullptr) {
                std::cout << "NewNgnEnc is be implemented!" <<
                (long)pNewNgnEnc->GetLow() <<
                (long)pNewNgnEnc->GetRange() <<
                (long)pNewNgnEnc->GetCode() << std::endl;
            } else if (pNewNgnDec != nullptr) {
                std::cout << "RngcDec is be implemented!" <<
                (long)pNewNgnDec->GetLow() <<
                (long)pNewNgnDec->GetRange() <<
                (long)pNewNgnDec->GetCode() << std::endl;
            } else {
                std::cout << "To be implemented!" << std::endl;
            }
        }
        EXPORTED int GetWrittenBits()
        {
            int retSize = 0;
            if (pAcEnc != nullptr) {
                retSize = pBitOut->GetSize();
                …
                retSize = pRngcEnc->GetSize();
            } else if (pNewNgnEnc != nullptr) {
                return pNewNgnEnc->GetSize();
                …
            return retSize * 8;
        }

###	Building library
To build lib the following lines need to be included to CMakeList [src/codec/entropy_coding/core_libs/cpp/CMakeLists.txt]

    file(GLOB_RECURSE SRCS_NEWNGN_CODING ${CMAKE_CURRENT_SOURCE_DIR}/coding_algos/new_ngn/*.cpp)
    message("SRCS_NEWNGN_CODING: ${SRCS_ NEWNGN_CODING}\n")

and

    target_sources(entropy_coding
        PUBLIC
        ${SRCS_BIT_STREAMS}
        ${SRCS_ARITHMETIC_CODING}
        ${SRCS_RANGE_CODING}
        ${SRCS_NEWNGN_CODING}
        ${SRCS_RANGEPAR_CODING}
        ${SRCS_PROB_MODELS}
        ${SRCS_CODING_MODULE}
    )

###	Setting up in configuration
Add the following argument to command line:

    -EC.type new_ngn

or to configuration:

    "EC": {
        "type": " new_ngn "
            },

The name should correspond to the name from section 'class ECInstanceFactory(object)'
###	Include calling of new engine into probability models
Source code related to probability models located in the following folder [src/codec/entropy_coding/core_libs/cpp/coding_module]
An example for Uniform Probability:

    extern NewNgnEncoder * pNewNgnEnc;
    extern NewNgnDecoder* pNewNgnDec;

    …
    EXPORTED void CompressUniform(const uint32_t *pData, const int dataSize, const int symbolNum)
    {
        …
        UniformProbModel probModel(symbolNum);
            if (g_coderType == 0) {
                pAcEnc->Write(probModel, static_cast<uint32_t>(symbol));
            else if (g_coderType == 111) {
                pNewNgnEnc->Write(probModel, static_cast<uint32_t>(symbol));
            } else {
                pRcEnc->Write(probModel, static_cast<uint32_t>(symbol));
            }
	    …
    }
    EXPORTED void DecompressUniform(int *pData, const int dataSize, const int symbolNum)
    {
        …
        UniformProbModel probModel(symbolNum);
            if (g_coderType == 0) {
                symbol = pAcDec->ReadEp(probModel);
            if (g_coderType == 111) {
                symbol = pNewNgnDec->ReadEp(probModel);
            } else {
                symbol = pRcDec->ReadEp(probModel);
            }
        …


##	Adding new probability model
###	class ECLibProxy(ECLibBase) and class ECLibCpp(ECLibBase):

Class ECLibProxy(ECLibBase) [src/codec/entropy_coding/lib_wrappers/proxy] is used for likelihood estimation for training and RDO purposes.
Class ECLibCpp(ECLibBase) [src/codec/entropy_coding/lib_wrappers/cpp] is used for inference encoding/decoding.

####	prepare new model

    class NewProbModel(BaseProbModel):
        def __init__(self, parameters of the model):

####	prepare new model wrapper
Prepare wrapper of new model in separate file located in `src/codec/entropy_coding/lib_wrappers/proxy`

    class NewProbWrapper(WrapperProbBase):
        def __init__(self, backend):

####	push the wrapper of new model to the list

        self.prob_wrappers = {
            'Agmm'   : AgmmProbWrapper(self.backend),
            'Asgm'   : AsgmProbWrapper(self.backend),
            'Custom' : CustomProbWrapper(self.backend),
            'Golomb' : GolombProbWrapper(self.backend),
            'Hist'   : HistProbWrapper(self.backend),
            'Uniform': UniformProbWrapper(self.backend),
            'New': NewProbWrapper(self.backend),
        }
####	Initialize IO

Add initialization of File IO or memory IO in `encode_init()`.
Add termination of the streaming in `encode_term()`.
##	Add appropriate method of entropy coding to ECModule [src/codec/entropy_coding/ec_module.py]

    class ECModule(nn.Module):
        def encode_New():
        def decode_New():
