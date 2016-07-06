#pragma once
#ifndef UTILITY_THREADING_H
#define UTILITY_THREADING_H

#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <memory>
#include <functional>
#include "Spin_System.h"
#include "Spin_System_Chain.h"

namespace Utility
{
	namespace Threading
	{
		extern std::map<std::shared_ptr<Data::Spin_System>, std::thread> llg_threads;
		extern std::map<std::shared_ptr<Data::Spin_System_Chain>, std::thread> gneb_threads;
		//static std::map<std::shared_ptr<Data::Spin_System_Chain_Collection>, std::thread> mmf_threads;
		
	}
}

#endif