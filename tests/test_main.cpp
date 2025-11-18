#include "MatrixTest.h"
#ifdef _WIN32
#include <windows.h>
#endif


int main() {
    #ifdef _WIN32
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    #endif

    std::cout << "*** Тест программы ***" << std::endl;
    
    MatrixTest::runAllTests();
    
    std::cout << "\n*** Тестирование завершено ***" << std::endl;
    
    return 0;
}
