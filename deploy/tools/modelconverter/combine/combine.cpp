#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include <stdlib.h>
#include<string.h>

#include <iostream>
#include <string>

#define BUFFER_SIZE 1024

int main(int argc, char **argv){
    if (argc < 3){
        std::cout << "USAGE:" << std::endl;
        std::cout << "./combine <out_file_path> <in_file1_path> <in_file2_path> " << std::endl;
        return -1;
    }

    std::string out_file = argv[1];
    std::string in_file1 = argv[2];
    std::string in_file2 = argv[3];
    std::cout << "In file: " << in_file1 << ", " << in_file2 << ", out file: " << out_file << std::endl;

    int fdw = open(out_file.c_str(), O_WRONLY | O_CREAT);
    if (-1 == fdw){
        return -1;
    }

    int fdr1 = open(in_file1.c_str(), O_RDONLY);
    if (-1 == fdr1){
        return -1;
    }

    int fdr2 = open(in_file2.c_str(), O_RDONLY);
    if (-1 == fdr2){
        return -1;
    }

    int fdr1_size = lseek(fdr1, 0L, SEEK_END);  
    lseek(fdr1, 0L, SEEK_SET);

    int fdr2_size = lseek(fdr2, 0L, SEEK_END);  
    lseek(fdr2, 0L, SEEK_SET);
    std::cout << "count: " << fdr1_size << ", " << fdr2_size << std::endl;

    char file_size_s[4];
    sprintf(file_size_s, "%d", fdr1_size);
    std::cout << "count: " << file_size_s  << std::endl;
    write(fdw, &fdr1_size, sizeof(int));

    char buf[BUFFER_SIZE];
    int count = 1;
    while(count){
        count = read(fdr1, buf, BUFFER_SIZE);
        if (-1 == count){
            printf("read error\n");
            close(fdw);
            close(fdr2);
            return -1;
        }
        write(fdw, buf, count);

        fdr1_size -= count;
        memset(buf, 0, BUFFER_SIZE);
    }

    count = 1;
    while(count){
        count = read(fdr2, buf, BUFFER_SIZE);
        if (-1 == count){
            printf("read error\n");
            close(fdw);
            close(fdr2);
            return -1;
        }
        write(fdw, buf, count);

        fdr2_size -= count;
        memset(buf, 0, BUFFER_SIZE);
    }

    close(fdw);
    close(fdr1);
    close(fdr2);

    return 0;
}