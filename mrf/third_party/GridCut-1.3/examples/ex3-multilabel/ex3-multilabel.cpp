/*
  ex3-multilabel
  ================

  This example shows how to do a simple multi-label segmentation with
  the help of an alpha-expansion solver based on GridCut.

*/

#include <cstdio>
#include <cmath>
#include <map>

#include <Image.h>
#include <AlphaExpansion/AlphaExpansion_2D_4C.h>

int main(int argc,char** argv)
{    
  const Image<float> image = imread<float>("image.png");
  const Image<RGB> scribbles = imread<RGB>("scribbles.png");
    
  const int width = image.width();
  const int height = image.height();
  
  int numLabels = 0;

  std::map<RGB,int> labelForColor;
  std::map<int,RGB> colorForLabel;

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
  {
    const RGB scribbleColor = scribbles(x,y);

    if (scribbleColor==RGB(1,1,1)) continue;

    if (labelForColor.count(scribbleColor)==0)
    {      
      const int label = numLabels;
    
      labelForColor[scribbleColor] = label;
      colorForLabel[label] = scribbleColor;      
      
      numLabels++;
    }    
  }

  int* dataCosts = new int[width*height*numLabels];

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
  {    
    for(int label=0;label<numLabels;label++)
    {      
      dataCosts[(x+y*width)*numLabels+label] = (colorForLabel[label]==scribbles(x,y) ? 0 : 1000);
    }
  }
 
  int** smoothnessCosts = new int*[width*height*2];

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
  {    
    const int xy = x+y*width;

    smoothnessCosts[xy*2+0] = new int[numLabels*numLabels];
    smoothnessCosts[xy*2+1] = new int[numLabels*numLabels];

    for(int label=0;label<numLabels;label++)
    for(int otherLabel=0;otherLabel<numLabels;otherLabel++)
    {
      #define WEIGHT(A) (1+1000*std::exp(-((A)*(A))/0.05))

      smoothnessCosts[xy*2+0][label+otherLabel*numLabels] = (x<width-1  && label!=otherLabel) ? WEIGHT(image(x+1,y)-image(x,y)) : 0;
      smoothnessCosts[xy*2+1][label+otherLabel*numLabels] = (y<height-1 && label!=otherLabel) ? WEIGHT(image(x,y+1)-image(x,y)) : 0;

      #undef WEIGHT
    }
  }
  

  typedef AlphaExpansion_2D_4C<int,int,int> Expansion; 

  Expansion* expansion = new Expansion(width,height,numLabels,dataCosts,smoothnessCosts);
  
  expansion->perform();
  
  int* labeling = expansion->get_labeling();
  

  Image<RGB> output(width,height);  

  for(int y=0;y<height;y++)
  for(int x=0;x<width;x++)
  {    
    output(x,y) = colorForLabel[labeling[x+y*width]]*image(x,y);
  }

  delete expansion;

  imwrite(output,"output.png");
  
  printf("The result was written to \"output.png\".\n");

  return 0;
}
