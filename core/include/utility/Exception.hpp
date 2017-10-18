#pragma once
#ifndef UTILITY_EXEPTION_H
#define UTILITY_EXEPTION_H

#include <Spirit/Exception.h>

namespace Utility
{
	enum class Exception
	{
		File_not_Found             = Exception_File_not_Found,
		System_not_Initialized     = Exception_System_not_Initialized,
		Division_by_zero           = Exception_Division_by_zero,
		Simulated_domain_too_small = Exception_Simulated_domain_too_small,
		Not_Implemented            = Exception_Not_Implemented,
        Non_existing_Image         = Exception_Non_existing_Image,
        Non_existing_Chain         = Exception_Non_existing_Chain,
        File_reading_error         = Exception_File_reading_error
        // TODO: from Chain.cpp
        // Last image deletion ?
        // Empty clipboard     ?
	};
  
  void Spirit_Exception( const Exception & ex, int idx_image=-1, int idx_chain=-1 );

  void Handle_Exception( int idx_image=-1, int idx_chain=-1 );
  
}

#endif