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
		Unknown_Exception          = Exception_Unknown_Exception,
        Non_existing_Image         = Exception_Non_existing_Image,
        Non_existing_Chain         = Exception_Non_existing_Chain
	};
  
  void Handle_exception( const Exception & ex, int idx_image, int idx_chain );
  
}

#endif