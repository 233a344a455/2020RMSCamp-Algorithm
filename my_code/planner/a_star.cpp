#include <iostream>
#include <time.h>
#include <algorithm>

using namespace std;

int * generateRandomMap()
{
    srand((unsigned)time(NULL));
    int rand_list[64];
    int map[8][8];
    for(int i = 0; i < 32; i++)
        rand_list[i] = rand_list[i+1] = rand() % 30 + 1;
    random_shuffle(rand_list, rand_list+64);
    for (int i = 0; i < 64; i++)
        map[i/8][i%8] = rand_list[i];
}

int main()
{
    int * map;
    map = generateRandomMap();
}