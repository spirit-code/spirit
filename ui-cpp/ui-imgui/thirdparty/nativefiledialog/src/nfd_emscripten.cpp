#include <emscripten.h>
#include <emscripten/html5.h>

#include "nfd_common.h"

#include <iostream>
#include <string>

EM_JS( char *, fs_read_log, (), { return FS.readFile( "Log.txt" ); } );

nfdresult_t NFD_OpenDialog( const nfdchar_t * filterList, const nfdchar_t * defaultPath, nfdchar_t ** outPath )
{
    nfdresult_t nfdResult = NFD_ERROR;
    NFDi_SetError( "Not implemented" );

    return nfdResult;
}

nfdresult_t
NFD_OpenDialogMultiple( const nfdchar_t * filterList, const nfdchar_t * defaultPath, nfdpathset_t * outPaths )
{
    nfdresult_t nfdResult = NFD_ERROR;
    NFDi_SetError( "Not implemented" );

    return nfdResult;
}

nfdresult_t NFD_SaveDialog( const nfdchar_t * filterList, const nfdchar_t * defaultPath, nfdchar_t ** outPath )
{
    nfdresult_t nfdResult = NFD_ERROR;
    NFDi_SetError( "Not implemented" );

    return nfdResult;
}

nfdresult_t NFD_PickFolder( const nfdchar_t * defaultPath, nfdchar_t ** outPath )
{
    nfdresult_t nfdResult = NFD_ERROR;
    NFDi_SetError( "Not implemented" );

    return nfdResult;
}
