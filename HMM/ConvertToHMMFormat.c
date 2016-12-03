#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>


int convert(char directory[])
{
  char* filename[2];
  filename[0] = "/training.txt";
  filename[1] = "/test.txt";
  
  for(int z=0; z<2; z++){
    char dir[300] = "";

    FILE *fp, *fptemp, *fpoutput;
    int c;
    int n = 0, tabCount = 0, intVal = 0, lineCount = 0;
    char label[20] = "";
    char sensorVal[12] = "";
    
    strcpy(dir,directory);
    strcat(dir,filename[z]);
    printf("input file: %s\n", dir);

    fp = fopen(dir,"r");

    fptemp = fopen("temp.txt","w+");

    char outputFile[300] = "";
    strcpy(outputFile,directory);

    if(z==0){
      strcat(outputFile,"/hmm_train.txt");      
      fpoutput = fopen(outputFile,"w+");
    } else if (z==1){
      strcat(outputFile,"/hmm_test.txt");      
      fpoutput = fopen(outputFile,"w+");
    } else {
      printf("Error!\n");
    }
    printf("output file: %s\n", outputFile);

    if(fp == NULL)
    {
      perror("Error in opening file: ");
      printf("Error in opening file: %s", dir);
      return(-1);
    }

    while(1){
      c = fgetc(fp);

      if(feof(fp)){
        break;
      }

      if(c == '\n'){
        tabCount = 0;
        fputc('\n',fptemp);
        lineCount = lineCount +1;
      }

      if(c=='\t'){
        tabCount = tabCount + 1;
      }

      if(tabCount >= 3){
        if(c=='\t'){
          if(tabCount == 15){
            fputc(c, fptemp);
          }
        } else {
          fputc(c, fptemp);
        }
      }
    }
    rewind(fptemp);
    int s = 0;

    while(s<lineCount){
      if(feof(fptemp)){
        break;
      }
      fscanf(fptemp, "%s %s", sensorVal, label);

      int sensorLength = 0;
      int sensorData = 0;
      int convertedLabel = 0;

      sensorLength = strlen(sensorVal);

      for(int i=0; i<sensorLength; i++){
        if(sensorVal[i]=='1'){
          sensorData = sensorData + pow(2,(sensorLength-(i+1)));
        }
      }

      //Matlab HMM does not accept emission with label '0'
      if(sensorData == 0){
        sensorData = 8192;
      }

      //Labels (ADLs included):  Leaving(1), Toileting(2), Showering(3), Sleeping(4), Breakfast(5), Lunch(6), Dinner(7), Snack(8), Spare_Time/TV(9), Grooming(10), Idle(11), Others(100)
      if(strcmp(label,"Leaving") == 0){
        convertedLabel = 1;
      } else if (strcmp(label,"Toileting") == 0){
        convertedLabel = 2;
      } else if (strcmp(label,"Showering") == 0){
        convertedLabel = 3;
      } else if (strcmp(label,"Sleeping") == 0){
        convertedLabel = 4;
      } else if (strcmp(label,"Breakfast") == 0){
        convertedLabel = 5;
      } else if (strcmp(label,"Lunch") == 0){
        convertedLabel = 6;
      } else if (strcmp(label,"Dinner") == 0){
        convertedLabel = 7;
      } else if (strcmp(label,"Snack") == 0){
        convertedLabel = 8;
      } else if (strcmp(label,"Spare_Time/TV") == 0){
        convertedLabel = 9;
      } else if (strcmp(label,"Grooming") == 0){
        convertedLabel = 10;
      } else if (strcmp(label,"Idle") == 0){
        convertedLabel = 11;
      } else {
        convertedLabel = 100;
      }
      fprintf(fpoutput, "%d %d\n", sensorData, convertedLabel);
      s++;
    }

    fclose(fp);
    fclose(fptemp);
    fclose(fpoutput);
  }
  return(0);
}


void main(){
  char* subDirectories[2];
  char* subsubDirectories[2];
  char* subsubsubDirectories[2];
  char* file[2];

  subDirectories[0] = "/OrdonezA";
  subDirectories[1] = "/OrdonezB";
  subsubDirectories[0] = "/with_idle_states";
  subsubDirectories[1] = "/without_idle_states";
  subsubsubDirectories[0] = "/Last_Sensor/";
  subsubsubDirectories[1] = "/Raw/";

  int result = 0;


  for(int j=0; j<2; j++){
    char mainDir[300] = "./../UCI-ADL-Binary-Dataset/Segmented_Dataset";
    strcat(mainDir, subDirectories[j]);
    for(int k=0; k<2; k++){
      char newDir[300] = "";
      strcpy(newDir,mainDir);
      strcat(newDir, subsubDirectories[k]);
      for(int l=0; l<2; l++){
        char newnewDir[300] = "";
        strcpy(newnewDir,newDir);
        strcat(newnewDir, subsubsubDirectories[l]);
        if(!(strcmp(subDirectories[j], "/OrdonezA"))){
          for(int m=1;m<15;m++){
            char finalDir[300] = "";
            strcpy(finalDir,newnewDir);
            char buffer[10];
            snprintf(buffer, 10, "%d", m);
            strcat(finalDir,buffer);
            result = convert(finalDir);
          }
        } else if (!(strcmp(subDirectories[j], "/OrdonezB"))){
          for(int m=1;m<22;m++){
            char finalDir[300] = "";
            strcpy(finalDir,newnewDir);
            char buffer[10];
            snprintf(buffer, 10, "%d", m);
            strcat(finalDir,buffer);
            result = convert(finalDir);
          }
        }
      }
    }
  }
}
