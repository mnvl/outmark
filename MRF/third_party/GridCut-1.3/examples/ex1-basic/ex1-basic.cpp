/*
  ex1-basic
  =========

  This example shows how to use GridCut to determine the
  minimum cut in the following 4-connected 2D grid graph:

                    (S) <- source
                     |
                    <8>
                     |
                     o--<5>--o--<1>--o
                     |       |       |
                    <2>     <3>     <4>
                     |       |       |
                     o--<4>--o--<5>--o
                     |       |       |
                    <1>     <2>     <6>
                     |       |       |
                     o--<7>--o--<8>--o
                                     |
                                    <9>
                                     |
                            sink -> (T)

  The number inside <> bracket denotes the edge's capacity.
  Capacities of edges between neighboring nodes are symmetric.

*/

#include <cstdio>
#include <GridCut/GridGraph_2D_4C.h>

int main(int argc, char **argv)
{
  typedef GridGraph_2D_4C<int,int,int> Grid;

  Grid* grid = new Grid(3,3);

  grid->set_terminal_cap(grid->node_id(0,0),8,0);
  grid->set_terminal_cap(grid->node_id(2,2),0,9);

  grid->set_neighbor_cap(grid->node_id(0,0),+1, 0,5);
  grid->set_neighbor_cap(grid->node_id(1,0),-1, 0,5);  
  grid->set_neighbor_cap(grid->node_id(0,0), 0,+1,2);  
  grid->set_neighbor_cap(grid->node_id(0,1), 0,-1,2);
  grid->set_neighbor_cap(grid->node_id(1,0),+1, 0,1);
  grid->set_neighbor_cap(grid->node_id(2,0),-1, 0,1);  
  grid->set_neighbor_cap(grid->node_id(1,0), 0,+1,3);  
  grid->set_neighbor_cap(grid->node_id(1,1), 0,-1,3);
  grid->set_neighbor_cap(grid->node_id(2,0), 0,+1,4);  
  grid->set_neighbor_cap(grid->node_id(2,1), 0,-1,4);
  grid->set_neighbor_cap(grid->node_id(0,1),+1, 0,4);
  grid->set_neighbor_cap(grid->node_id(1,1),-1, 0,4);  
  grid->set_neighbor_cap(grid->node_id(0,1), 0,+1,1);  
  grid->set_neighbor_cap(grid->node_id(0,2), 0,-1,1);
  grid->set_neighbor_cap(grid->node_id(1,1),+1, 0,5);
  grid->set_neighbor_cap(grid->node_id(2,1),-1, 0,5);  
  grid->set_neighbor_cap(grid->node_id(1,1), 0,+1,2);  
  grid->set_neighbor_cap(grid->node_id(1,2), 0,-1,2);
  grid->set_neighbor_cap(grid->node_id(2,1), 0,+1,6);  
  grid->set_neighbor_cap(grid->node_id(2,2), 0,-1,6);  
  grid->set_neighbor_cap(grid->node_id(0,2),+1, 0,7);  
  grid->set_neighbor_cap(grid->node_id(1,2),-1, 0,7);  
  grid->set_neighbor_cap(grid->node_id(1,2),+1, 0,8);  
  grid->set_neighbor_cap(grid->node_id(2,2),-1, 0,8);  

  grid->compute_maxflow();

  printf("Min-cut partition:\n");

  for(int y=0;y<3;y++)
  {  
    for(int x=0;x<3;x++)  
    {
      printf("%c",(grid->get_segment(grid->node_id(x,y)) == 0) ? 'S' : 'T');
    }
    printf("\n");
  }
  
  delete grid;

  return 0;
}