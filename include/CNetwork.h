#ifndef CNETWORK_H
#define CNETWORK_H

#include <stdint.h>



/////////////////////////////////////////////////////////////////////////////////
//                               Centrality                                    //
/////////////////////////////////////////////////////////////////////////////////

void deg_in(long double **ad_mat, long double *property);
void deg_out(long double **ad_mat,long double *property);
void Betweenness_directed(long double **ad_mat,long double *property);
void harmonic_closenness_in(long double **ad_mat,long double *property);
void harmonic_closenness_out(long double **ad_mat,long double *property);
void leverage_in(long double **ad_mat,long double *property);
void leverage_out(long double **ad_mat,long double *property);
void PageRank_in(long double **ad_mat,long double *property);
void PageRank_out(long double **ad_mat,long double *property);
void Random_Walk_Betweenness_directed(long double **ad_mat,long double *property,long int max_steps,long int MAX_WALKS);
void Harmonic_Random_Walk_Closeness_in(long double **ad_mat,long double *property,long int max_steps,long int MAX_WALKS);
void Harmonic_Random_Walk_Closeness_out(long double **ad_mat,long double *property,long int max_steps,long int MAX_WALKS);



/////////////////////////////////////////////////////////////////////////////////
//                             Network Equal                                   //
/////////////////////////////////////////////////////////////////////////////////

void Erdos_Renyi_directed_P(long double **ad_mat, long double p);
void Erdos_Renyi_directed_EQ(long double **ad_mat,long int E);
void Erdos_Renyi_EQ(long double **ad_mat,long int E);
void Bollobas_Borgs_Chayes_Riordan_EQ(long double **ad_mat,double alpha,double beta,double gamma,double delta_in,double delta_out);
void Bollobas_Borgs_Chayes_Riordan_undirect_EQ(long double **ad_mat,double alpha,double beta,double delta);
void Watts_Strogatz_directed_EQ(long double **ad_mat,double p_ws,double Average_Degree_out);
void Watts_Strogatz_EQ(long double **ad_mat,double p_ws,double Average_Degree);
static double edge_probability(int i, int j, int L, double p0, double beta);
static int ring_distance(int i, int j, int L);
void Configurational_model_directed_EQ(long double **ad_mat_real, long double **ad_mat_copy, long int E);
void Configurational_model_directed_percent(long double **ad_mat_real, long double **ad_mat_copy, long int E,long int percent);
void Configurational_model_undirected_EQ(long double **ad_mat_real, long double **ad_mat_copy, long int E);
void duplication_divergence_directed(long double **A,int n_target, double p_keep, double p_parent);
void duplication_divergence_undirected(long double **A,int n_target, double p_keep, double p_parent);
void barabasi_albert_undirected(long double **ad_mat, long int N, long int m_0, long int m);

/////////////////////////////////////////////////////////////////////////////////
//                               properties                                    //
/////////////////////////////////////////////////////////////////////////////////


void Copy_Network(long double **ad_mat,long double ** ad_mat_real);
double shannon_entropy(long double *property);
void compute_histogram(long double *data, int data_size, long int *bins,double Number_Of_Bins, double min_val, double max_val);
void empty_mat(long double **ad_mat);
void normalize(long double *property,long int n);
void empty_vec(long double * property );
void Pdf(long double *property,long int *bins,double Number_Of_Bins,long int n);
void is_zero_out_degree(long double **ad_mat);
void generateLineDigraph(long double** A, int E, long double **A_L);
void generateLineGraph(long double** A, int E, long double **A_L);
void VisualizeGraph(long double **ad_mat,long int N,const char *output_file,const char *node_color,const char *edge_color,double node_size,double edge_width,int directed);
void DirectedClusteringCoefficient(long double **A, long int N,long double *C);
void NodePropertySpectrum(long double **A, long int N,long double *Ci, const char *mode,long double *Ck);
void ClusteringCoefficientUndirected(long double **A, long int N, long double *C);
void add_edge(long double **A, long int u, long int v);
static void add_arc(long double **A, long int u, long int v);
static double rand01(void);
long double Average(long double *v,long int size);
void ComputeKNN_Directed(long double **A,long int N,long double *knn,const char *neighbor_dir,const char *degree_dir);
long double standard_deviation(long double *Network,long double average);
long int max_core_number(long double **ad_mat, long int n);
long int directed_max_k_core(long double **ad_mat, long int n, const char *mode);
void make_directed(long double **ad_mat,long int n,long double p_bidirection);
long double balance_index(long double **ad_mat, long int n);
long double heterogeneity(long double **A, long int N, char mode[]);
int WeaklyConnected(long double **ad_mat);
#endif