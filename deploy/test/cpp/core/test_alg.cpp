#include "core/alg.h"

int main()
{
    AlgManager *algm = AlgManager::Instance();
    AlgCreator *alg_creator = algm->GetAlgCreator("detect", 'yolo');

    AlgPtr alg = alg_creator->Create();

    

    return 0;
}