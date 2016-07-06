#pragma once
#ifndef UTILITY_EXEPTION_H
#define UTILITY_EXEPTION_H

namespace Utility
{
	enum class Exception
	{
		File_not_Found,
		System_not_Initialized,
		Division_by_zero,
		Simulated_domain_too_small,
		Not_Implemented,
		Unknown_Exception
	};
}

#endif