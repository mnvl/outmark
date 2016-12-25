
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/push_relabel_max_flow.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/graph/edmonds_karp_max_flow.hpp>
#include <iostream>
#include <string>

static const int SIDE = 200;
static const int TOTAL_VERTICES = SIDE * SIDE * SIDE;

inline long vnum(int i, int j, int k) {
    return ((long(i) * SIDE) + long(j)) * SIDE + long(k);
}

int main() {
  using namespace boost;

  typedef adjacency_list_traits<vecS, vecS, directedS> Traits;
  typedef adjacency_list<
      vecS, vecS, directedS, property<vertex_name_t, long>,
      property<edge_capacity_t, float,
               property<edge_residual_capacity_t, float,
                        property<edge_reverse_t, Traits::edge_descriptor>>>>
      Graph;
  typedef Graph::edge_descriptor Edge;

  Graph g;
  long flow;

  property_map<Graph, edge_capacity_t>::type capacity = get(edge_capacity, g);
  property_map<Graph, edge_reverse_t>::type rev = get(edge_reverse, g);
  property_map<Graph, edge_residual_capacity_t>::type residual_capacity =
      get(edge_residual_capacity, g);

  Traits::vertex_descriptor s, t;

  std::cout << "building vertices\n";
  for (int i = 0; i < SIDE; ++i) {
      for (int j = 0; j < SIDE; ++j) {
          for (int k = 0; k < SIDE; ++k) {
              add_vertex(vnum(i, j, k),g);
          }
      }
  }

  s = add_vertex(TOTAL_VERTICES + 1, g);
  t = add_vertex(TOTAL_VERTICES + 2, g);

  std::cout << "building edges\n";
  for (int i = 1; i < SIDE - 1; ++i) {
      for (int j = 1; j < SIDE - 1; ++j) {
          for (int k = 1; k < SIDE - 1; ++k) {
              add_edge(vnum(i - 1, j, k), vnum(i, j, k), edge_capacity_t(10.0f), g);
              add_edge(vnum(i + 1, j, k), vnum(i, j, k), edge_capacity_t(10.0f), g);
              add_edge(vnum(i, j - 1, k), vnum(i, j, k), edge_capacity_t(10.0f), g);
              add_edge(vnum(i, j + 1, k), vnum(i, j, k), edge_capacity_t(10.0f), g);
              add_edge(vnum(i, j, k - 1), vnum(i, j, k), edge_capacity_t(10.0f), g);
              add_edge(vnum(i, j, k + 1), vnum(i, j, k), edge_capacity_t(10.0f), g);
              add_edge(TOTAL_VERTICES + 1, vnum(i, j, k), edge_capacity_t(10.0f), g);
              add_edge(vnum(i, j, k), TOTAL_VERTICES + 2, edge_capacity_t(10.0f), g);
          }
      }
  }

  std::cout << "max flow...\n";
  flow = push_relabel_max_flow(g, s, t);
  //flow =  boykov_kolmogorov_max_flow(g ,s, t);
  //flow = edmonds_karp_max_flow(g, s, t);

  std::cout << "c  The total flow:" << std::endl;
  std::cout << "s " << flow << std::endl << std::endl;

  /*
  std::cout << "c flow values:" << std::endl;
  graph_traits<Graph>::vertex_iterator u_iter, u_end;
  graph_traits<Graph>::out_edge_iterator ei, e_end;
  for (boost::tie(u_iter, u_end) = vertices(g); u_iter != u_end; ++u_iter)
    for (boost::tie(ei, e_end) = out_edges(*u_iter, g); ei != e_end; ++ei)
      if (capacity[*ei] > 0)
        std::cout << "f " << *u_iter << " " << target(*ei, g) << " "
                  << (capacity[*ei] - residual_capacity[*ei]) << std::endl;
  */
  return 0;
}
