/*-------------------------------------------------------------------------------
 This file is part of ranger.

 Copyright (c) [2014-2018] [Marvin N. Wright]

 This software may be modified and distributed under the terms of the MIT license.

 Please note that the C++ core of ranger is distributed under MIT license and the
 R package "ranger" under GPL3 license.
 #-------------------------------------------------------------------------------*/

#ifndef FOREST_H_
#define FOREST_H_

#include <vector>
#include <iostream>
#include <random>
#include <ctime>
#include <memory>
#ifndef OLD_WIN_R_BUILD
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#endif

#include "globals.h"
#include "Tree.h"
#include "Data.h"

namespace ranger {
class Forest {
public:
  Forest();
  Forest(const Forest&) = delete;
  Forest& operator=(const Forest&) = delete;

  virtual ~Forest() = default;

  // Init from c++ main or Rcpp from R
  void initCpp(std::string dependent_variable_name, MemoryMode memory_mode, std::string input_file, uint mtry,
      std::string output_prefix, uint num_trees, std::ostream* verbose_out, uint seed, uint num_threads,
      std::string load_forest_filename, ImportanceMode importance_mode, uint min_node_size,
      std::string split_select_weights_file, const std::vector<std::string>& always_split_variable_names,
      std::string status_variable_name, bool sample_with_replacement,
      const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      std::string case_weights_file, bool predict_all, double sample_fraction, double alpha, double minprop,
      bool holdout, PredictionType prediction_type, uint num_random_splits, uint max_depth,
      const std::vector<double>& regularization_factor, bool regularization_usedepth);
  void initR(std::unique_ptr<Data> input_data, uint mtry, uint num_trees, std::ostream* verbose_out, uint seed,
      uint num_threads, ImportanceMode importance_mode, uint min_node_size,
      std::vector<std::vector<double>>& split_select_weights,
      const std::vector<std::string>& always_split_variable_names, bool prediction_mode, bool sample_with_replacement,
      const std::vector<std::string>& unordered_variable_names, bool memory_saving_splitting, SplitRule splitrule,
      std::vector<double>& case_weights, std::vector<std::vector<size_t>>& manual_inbag, bool predict_all,
      bool keep_inbag, std::vector<double>& sample_fraction, double alpha, double minprop, bool holdout,
      PredictionType prediction_type, uint num_random_splits, bool order_snps, uint max_depth,
      const std::vector<double>& regularization_factor, bool regularization_usedepth);
  void init(std::unique_ptr<Data> input_data, uint mtry, std::string output_prefix,
      uint num_trees, uint seed, uint num_threads, ImportanceMode importance_mode, uint min_node_size,
      bool prediction_mode, bool sample_with_replacement, const std::vector<std::string>& unordered_variable_names,
      bool memory_saving_splitting, SplitRule splitrule, bool predict_all, std::vector<double>& sample_fraction,
      double alpha, double minprop, bool holdout, PredictionType prediction_type, uint num_random_splits,
      bool order_snps, uint max_depth, const std::vector<double>& regularization_factor, bool regularization_usedepth);
  virtual void initInternal() = 0;

  // Grow or predict
  void run(bool verbose, bool compute_oob_error);

  // Write results to output files
  void writeOutput();
  virtual void writeOutputInternal() = 0;
  virtual void writeConfusionFile() = 0;
  virtual void writePredictionFile() = 0;
  void writeImportanceFile();

  // Save forest to file
  void saveToFile();
  virtual void saveToFileInternal(std::ofstream& outfile) = 0;

  std::vector<std::vector<std::vector<size_t>>> getChildNodeIDs() {
    std::vector<std::vector<std::vector<size_t>>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getChildNodeIDs());
    }
    return result;
  }
  std::vector<std::vector<size_t>> getSplitVarIDs() {
    std::vector<std::vector<size_t>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getSplitVarIDs());
    }
    return result;
  }
  std::vector<std::vector<double>> getSplitValues() {
    std::vector<std::vector<double>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getSplitValues());
    }
    return result;
  }
  const std::vector<double>& getVariableImportance() const {
    return variable_importance;
  }
  const std::vector<double>& getVariableImportanceCasewise() const {
    return variable_importance_casewise;
  }
  double getOverallPredictionError() const {
    return overall_prediction_error;
  }
  const std::vector<std::vector<std::vector<double>>>& getPredictions() const {
    return predictions;
  }
  size_t getNumTrees() const {
    return num_trees;
  }
  uint getMtry() const {
    return mtry;
  }
  uint getMinNodeSize() const {
    return min_node_size;
  }
  size_t getNumIndependentVariables() const {
    return num_independent_variables;
  }

  const std::vector<bool>& getIsOrderedVariable() const {
    return data->getIsOrderedVariable();
  }

  std::vector<std::vector<size_t>> getInbagCounts() const {
    std::vector<std::vector<size_t>> result;
    for (auto& tree : trees) {
      result.push_back(tree->getInbagCounts());
    }
    return result;
  }

  const std::vector<std::vector<size_t>>& getSnpOrder() const {
    return data->getSnpOrder();
  }

protected:
  void grow();
  virtual void growInternal() = 0;

  // Predict using existing tree from file and data as prediction data
  void predict();
  virtual void allocatePredictMemory() = 0;
  virtual void predictInternal(size_t sample_idx) = 0;

  void computePredictionError();
  virtual void computePredictionErrorInternal() = 0;

  void computePermutationImportance();

  // Multithreading methods for growing/prediction/importance, called by each thread
  void growTreesInThread(uint thread_idx, std::vector<double>* variable_importance);
  void predictTreesInThread(uint thread_idx, const Data* prediction_data, bool oob_prediction);
  void predictInternalInThread(uint thread_idx);
  void computeTreePermutationImportanceInThread(uint thread_idx, std::vector<double>& importance,
      std::vector<double>& variance, std::vector<double>& importance_casewise);

  // Load forest from file
  void loadFromFile(std::string filename);
  virtual void loadFromFileInternal(std::ifstream& infile) = 0;
  void loadDependentVariableNamesFromFile(std::string filename);

  // Load data from file
  std::unique_ptr<Data> loadDataFromFile(const std::string& data_path);

  // Set split select weights and variables to be always considered for splitting
  void setSplitWeightVector(std::vector<std::vector<double>>& split_select_weights);
  void setAlwaysSplitVariables(const std::vector<std::string>& always_split_variable_names);

  // Show progress every few seconds
#ifdef OLD_WIN_R_BUILD
  void showProgress(std::string operation, clock_t start_time, clock_t& lap_time);
#else
  void showProgress(std::string operation, size_t max_progress);
#endif

  // Verbose output stream, cout if verbose==true, logfile if not
  std::ostream* verbose_out;

  std::vector<std::string> dependent_variable_names; // time,status for survival
  size_t num_trees;
  uint mtry;
  uint min_node_size;
  size_t num_independent_variables;
  uint seed;
  size_t num_samples;
  bool prediction_mode;
  MemoryMode memory_mode;
  bool sample_with_replacement;
  bool memory_saving_splitting;
  SplitRule splitrule;
  bool predict_all;
  bool keep_inbag;
  std::vector<double> sample_fraction;
  bool holdout;
  PredictionType prediction_type;
  uint num_random_splits;
  uint max_depth;

  // MAXSTAT splitrule
  double alpha;
  double minprop;

  // Multithreading
  uint num_threads;
  std::vector<uint> thread_ranges;
#ifndef OLD_WIN_R_BUILD
  std::mutex mutex;
  std::condition_variable condition_variable;
#endif

  std::vector<std::unique_ptr<Tree>> trees;
  std::unique_ptr<Data> data;

  std::vector<std::vector<std::vector<double>>> predictions;
  double overall_prediction_error;

  // Weight vector for selecting possible split variables, one weight between 0 (never select) and 1 (always select) for each variable
  // Deterministic variables are always selected
  std::vector<size_t> deterministic_varIDs;
  std::vector<std::vector<double>> split_select_weights;

  // Bootstrap weights
  std::vector<double> case_weights;

  // Pre-selected bootstrap samples (per tree)
  std::vector<std::vector<size_t>> manual_inbag;

  // Random number generator
  std::mt19937_64 random_number_generator;

  std::string output_prefix;
  ImportanceMode importance_mode;

  // Regularization
  std::vector<double> regularization_factor;
  bool regularization_usedepth;
  std::vector<bool> split_varIDs_used;
  
  // Variable importance for all variables in forest
  std::vector<double> variable_importance;

  // Casewise variable importance for all variables in forest
  std::vector<double> variable_importance_casewise;

  // Computation progress (finished trees)
  size_t progress;
#ifdef R_BUILD
  size_t aborted_threads;
  bool aborted;
#endif
};

class SkyrmionRaceTrack {
public:
  // ----------------------------------------below is my own code --------------------------------------------
  // ----------------------------------------below is shift code ---------------------------------------------
  vector<vector<double>> memory, parallel_memory, parallel_writing_buffer ;
  vector<vector<int>> ap_index, parallel_ap_index, tree_scope ;
  vector<vector<int>> parallel_prenode, parallel_track_info, parallel_access_time ;
  vector<vector<int>> parallel_prenode_access_time ;
  vector<int> p_tree_scope, p_tree_scope_end, p_access_port_start ;
  vector<int> p_access_port_end, p_track_shift, p_track_at ;
  vector<vector<long long int>> thread_handles_tree, ap_access_time ;
  vector<long long int> thread_write_shift_count, thread_read_shift_count, thread_access_time ;
  int total_track = 0, shift_count = 0, word_nums = 0 ;
  int ap_nums = 0, last_index = 0, remainder_count = 0 ;
  int last_apindex, parallel_total_track = 0, finish = 0 ;
  int total_reading_shift_distance = 0, access_time = 0, parallel_access = 0 ;
  int tree_shared_track_count = 0, p_w_s = 0 ; // To record each tree need to have how many tracks? 
  long long int total_access_time = 0 ;
  bool NLF_Mode ;
  int space = 0 ;

  void Refresh( vector<double> &input ) {
    for ( int i = 0 ; i < input.size() ; i++ ) {
      if ( i % ( word_nums + 1 ) != 0 ) 
        input[i] = -1 ; // -1 means empty space
      else 
        input[i] = -100 ; // -100 means access port
    } // for
  } // Refresh

  void Initialize( int words, int aps ) {
    word_nums = words ;
    ap_nums = aps ;
  } // Initialize()

  int Thread_ShiftP( int pos, int &s_c ) { // Since the parallel reading, we can't use the orginal ShiftP for protecting global number "shift_count".
    s_c++ ;
    for ( int i = 0 ; i < ap_index[pos].size() ; i++ )
      ap_index[pos][i]++ ;
  } // Thread_ShiftP()

  int Thread_ShiftN( int pos, int &s_c ) { // Since the parallel reading, we can't use the orginal ShiftN for protecting global number "shift_count".
    s_c++ ;
    for ( int i = 0 ; i < ap_index[pos].size() ; i++ )
      ap_index[pos][i]-- ;
  } // Thread_ShiftN()

  void ShiftP( int pos ) {
    shift_count++ ;
    for ( int i = 0 ; i < ap_index[pos].size() ; i++ )
      ap_index[pos][i]++ ;
  } // ShiftP()

  void ShiftN( int pos ) {
    shift_count++ ;
    for ( int i = 0 ; i < ap_index[pos].size() ; i++ )
      ap_index[pos][i]-- ;
  } // ShiftN()

  void P_ShiftP( int pos ) {
    shift_count++ ;
    for ( int i = 0 ; i < parallel_ap_index[pos].size() ; i++ ) 
      parallel_ap_index[pos][i]++ ;
  } // P_ShiftP()

  void P_ShiftN( int pos ) {
    shift_count++ ;
    for ( int i = 0 ; i < parallel_ap_index[pos].size() ; i++ )
      parallel_ap_index[pos][i]-- ;
  } // P_ShiftN()

  void Addtrack() {
    int id = 0 ;
    vector<double> track ;
    vector<int> index_table ;
    track.resize( ap_nums * word_nums + ap_nums ) ;
    Refresh( track ) ;
    memory.push_back( track ) ;
    index_table.resize( ap_nums ) ;
    ap_index.push_back( index_table ) ;
    total_track++ ;
    for ( int i = 0 ; i < track.size() ; i++ ) {
      if ( i % ( word_nums + 1 ) == 0 ) {
        ap_index[total_track-1][id] = i ;
        id++ ;
      } // if
    } // for
  } // Addtrack()

  void P_Addtrack() {
    int id = 0 ;
    vector<double> track ;
    vector<int> index_table ;
    vector<long long int> counter_table ;
    track.resize( ap_nums * word_nums + ap_nums ) ;
    Refresh( track ) ;
    parallel_memory.push_back( track ) ;
    index_table.resize( ap_nums ) ;
    counter_table.resize( ap_nums ) ;
    parallel_ap_index.push_back( index_table ) ;
    ap_access_time.push_back( counter_table ) ;
    parallel_total_track++ ;
    for ( int i = 0 ; i < track.size() ; i++ ) {
      if ( i % ( word_nums + 1 ) == 0 ) {
        parallel_ap_index[parallel_total_track-1][id] = i ;
        id++ ;
      } // if
    } // for
  } // P_Addtrack()

  void PrintSfift() {
    cout << "Thid is parallel writing shift: " << p_w_s << endl ;
    cout << "This is shift : " << shift_count << endl ;
    cout << "This is track count : " << total_track << endl ;
  } // PrintSfift()

  void Print() {
    for ( int i = 0 ; i < memory.size() ; i++ ) {
      cout << "Track" << i << ": " ;
      for ( int j = 0 ; j < memory[i].size() ; j++ )
        cout << memory[i][j] << ", " ;
      cout << endl ;
    } // for
  } // Print()

  void P_Print() {
    for ( int i = 0 ; i < parallel_memory.size() ; i++ ) {
      cout << "Track" << i << ": " ;
      for ( int j = 0 ; j < parallel_memory[i].size() ; j++ )
        cout << parallel_memory[i][j] << ", " ;
      cout << endl ;
    } // for
  } // P_Print()

  void BacktoRootCompress( int track_start, int track_end, int nums_start ) {
    int limit ;
    for ( int i = track_start ; i <= track_end ; i++ ) {
      if ( i == track_start ) 
        limit = nums_start ;
      else 
        limit = 0 ;

      while ( ap_index[i][0] < limit ) 
        ShiftP(i) ;
      while ( ap_index[i][0] > limit ) 
        ShiftN(i) ; 
    } // for
  } // BacktoRootCompress() ;

  void BacktoRoot() {
    for( int i = total_track - 1 ; i >= 0 ; i-- ) {
      while( ap_index[i][0] > 0 ) // Just check whether the first access port of fist track is on the location 0 or not.
        ShiftN(i) ;
    } // for
  } // BacktoRoot()

  int ThreadBacktoRoot(int id, int &s_c) {
    for( int i = 0 ; i < tree_scope[id].size() ; i++ ) {
      while( ap_index[tree_scope[id][i]][0] > 0 ) // Just check whether the first access port of fist track is on the location 0 or not.
        Thread_ShiftN(tree_scope[id][i], s_c) ;
    } // for
  } // ThreadBacktoRoot()

  void P_BacktoRoot(bool ct) { // ct is used for determining whether the shift of backing to root need to be caculated.
    for( int i = parallel_total_track - 1 ; i >= 0 ; i-- ) {
      while( parallel_ap_index[i][0] > 0 ) { // Just check whether the first access port of fist track is on the location 0 or not.
        P_ShiftN(i) ;
        p_w_s++ ;
        if ( ct )
          p_track_shift[i]++ ;
      } // while
    } // for
  } // P_BacktoRoot()

  /* void Parallel_Write( int tree_id, vector<double> &buffer ) { // **********success code**********
    int track_id = finish / num_threads, skip = ap_nums / num_threads ;
    int pos = (finish % num_threads) * skip, remainder = pos ;

    if ( track_id >= parallel_memory.size() ) {
      P_Addtrack() ;
      vector<int> info ;
      parallel_track_info.push_back(info) ;
    } // if

    parallel_track_info[parallel_track_info.size()-1].push_back(tree_id) ;
    for ( int i = 0 ; i < buffer.size() ; i++ ) {
      if ( remainder == pos + skip )
        remainder = pos ;

      if ( parallel_memory[parallel_total_track-1][parallel_ap_index[parallel_total_track-1][remainder]] != -1 ) 
        if ( parallel_ap_index[parallel_total_track-1][remainder] + 1 < parallel_memory[0].size() &&
             parallel_memory[parallel_total_track-1][parallel_ap_index[parallel_total_track-1][remainder]+1] == -1 )
          P_ShiftP(parallel_total_track-1) ; 

      if ( i == 0 ) {
        p_tree_scope[tree_id] = parallel_total_track-1 ;
        p_access_port_start[tree_id] = remainder ;
      } // if

      parallel_memory[parallel_total_track-1][parallel_ap_index[parallel_total_track-1][remainder]] = buffer[i] ;
      remainder++ ;
    } // for

    p_tree_scope_end[tree_id] = parallel_total_track-1 ;
    p_access_port_end[tree_id] = remainder ;
    buffer.clear() ;
    P_BacktoRoot() ;
  } // Parallel_Write() */

  /* void Parallel_Write( int tree_id, vector<double> &buffer ) { // ******************version 2*******************
    int threshold = finish / num_threads * tree_shared_track_count, skip = ap_nums / num_threads ;
    int pos = (finish % num_threads) * skip, remainder = pos, now_track ;
    int iterator = 1 ;

    if ( threshold >= parallel_memory.size() ) {
      for ( int i = 0 ; i < tree_shared_track_count ; i++ ) {
        P_Addtrack() ;
        p_track_shift.push_back(0) ;
        vector<int> info ;
        parallel_track_info.push_back(info) ;
      } // for
    } // if

    now_track = parallel_memory.size() - tree_shared_track_count ;
    for ( int i = parallel_track_info.size()-1 ; i >= now_track ; i-- ) {
      parallel_track_info[i].push_back(tree_id) ;
    } // for

    for ( int i = 0 ; i < buffer.size() ; i++ ) { 
      // cout << buffer[i] << endl ;
      if ( i >= PR_placement_limit * iterator ) {
        iterator++ ;
        now_track++ ;
      } // if
        
      if ( remainder == pos + skip )
        remainder = pos ;

      if ( parallel_memory[now_track][parallel_ap_index[now_track][remainder]] != -1 ) 
        if ( parallel_ap_index[now_track][remainder] + 1 < parallel_memory[0].size() &&
             parallel_memory[now_track][parallel_ap_index[now_track][remainder]+1] == -1 ) {
          P_ShiftP(now_track) ; 
          p_w_s++ ;
      } // if

      if ( i == 0 ) {
        p_tree_scope[tree_id] = now_track ;
        p_access_port_start[tree_id] = remainder ;
      } // if

      parallel_memory[now_track][parallel_ap_index[now_track][remainder]] = buffer[i] ;
      remainder++ ;
    } // for

    p_tree_scope_end[tree_id] = parallel_total_track-1 ;
    p_access_port_end[tree_id] = remainder ;
    buffer.clear() ;
    P_BacktoRoot(false) ;
  } // Parallel_Write() */

  void Parallel_Write(int num_threads) { // ********************version 3************************
    vector<int> now_tree_index(parallel_writing_buffer.size(), 0) ;
    int a_group_of_ap = ap_nums / num_threads, now_tree = 0, iteration = parallel_writing_buffer.size() ;
    int last = parallel_writing_buffer.size()-1, time = 0 ;
    bool need_to_plus = true ;

    while ( iteration > 0 ) {
      P_Addtrack() ;
      vector<int> info ;
      parallel_track_info.push_back(info) ;
      p_track_shift.push_back(0) ;
      P_ShiftP( parallel_total_track-1 ) ;
      p_w_s++ ;
      for ( int i = 0 ; i < word_nums ; i++ ) {
        for ( int j = (time % num_threads) * a_group_of_ap ; j < ap_nums ; j++ ) {
          if ( j != 0 && j % a_group_of_ap == 0 && need_to_plus ) 
            now_tree++ ; 

          if ( now_tree_index[now_tree] == 0 ) {
            p_tree_scope[now_tree] = parallel_total_track-1 ;
            p_access_port_start[now_tree] = j ;
          } // if
       
          if ( now_tree < parallel_writing_buffer.size() && now_tree_index[now_tree] < parallel_writing_buffer[now_tree].size() ) {
            parallel_memory[parallel_total_track-1][parallel_ap_index[parallel_total_track-1][j]] = parallel_writing_buffer[now_tree][now_tree_index[now_tree]] ;
            now_tree_index[now_tree]++ ;
            space++ ;

            if ( now_tree_index[now_tree] == parallel_writing_buffer[now_tree].size() )
              iteration-- ;
          } // if

          need_to_plus = true ;
        } // for
        now_tree = 0 ;
        time = 0 ;

        while ( now_tree_index[now_tree] >= parallel_writing_buffer[now_tree].size() ) {
          now_tree++ ;
          time++ ;
          need_to_plus = false ;
        } // while

        if ( now_tree_index[now_tree] == 0 ) 
          break ;

        if ( i < word_nums-1 ) {
          P_ShiftP( parallel_total_track-1 ) ;
          p_w_s++ ;
        } // if
      } // for
    } // while
  } // Parallel_Write()

  void Level_Tree_Write( vector<double> input, int index ) {
    int remainder ;
    tree_scope[index].push_back(total_track-1) ;
    for ( int i = 0 ; i < input.size() ; i++ ) {
      remainder = i % ap_nums ;
      if ( memory[total_track-1][ap_index[total_track-1][remainder]] != -1 ) {
        if ( ap_index[total_track-1][remainder] + 1 < memory[0].size() &&
             memory[total_track-1][ap_index[total_track-1][remainder]+1] == -1 )
          ShiftP(total_track-1) ;
        else {
          Addtrack() ;
          tree_scope[index].push_back(total_track-1) ;
          ShiftP(total_track-1) ;
        } // else
      } // if
      memory[total_track-1][ap_index[total_track-1][remainder]] = input[i] ;
    } // for

    BacktoRoot() ;
  } // Level_Tree_Write
  
  /* void Level_Tree_Write( vector<double> input, int index ) { // FOR COMPACT ONLY
    int remainder ;
    for ( int i = 0 ; i < input.size() ; i++ ) {
      remainder = remainder_count % ap_nums ;
      if ( remainder == 0 )
        remainder_count = 0 ;

      if ( memory[total_track-1][ap_index[total_track-1][remainder]] != -1 ) {
        if ( ap_index[total_track-1][remainder] + 1 < memory[0].size() &&
             memory[total_track-1][ap_index[total_track-1][remainder]+1] == -1 )
          ShiftP(total_track-1) ;
        else {
          Addtrack() ;
          ShiftP(total_track-1) ;
        } // else
      } // if   

      if ( i == 0 ) {
        tree_scope[index] = total_track-1 ;
        access_port_start[index] = remainder ;
        wordnum_start[index] = (ap_index[total_track-1][remainder] % (word_nums+1)) ;
      } // if

      memory[total_track-1][ap_index[total_track-1][remainder]] = input[i] ;
      remainder_count++ ;
    } // for

    tree_scope_end[index] = total_track-1 ;
    access_port_end[index] = remainder ;
    wordnum_end[index] = (ap_index[total_track-1][remainder] % (word_nums+1)) ;
  } // Level_Tree_Write */ // FOR COMPACT ONLY
  
  vector<double> Non_Leaf_First_Sort( vector<double> input, vector<double> parent, vector<int> nodeid ) {
    vector<double> temp ;
    vector<bool> table ( input.size(), true ) ;
    for ( int i = 0 ; i < parent.size() ; i++ ) 
      temp.push_back( parent[i] ) ;
    
    for ( int i = 0 ; i < nodeid.size() ; i++ ) 
      table[nodeid[i]] = false ;

    for ( int i = 0 ; i < input.size() ; i++ ) {
      if ( table[i] )
        temp.push_back( i ) ;
    } // for
    
    return temp ;
  } // Non_Leaf_First()

  /* int Parallel_Read( double num, int index, int &a_t ) {
    bool out = false ;
    int count ;
    shift_count = 0 ;

    count = parallel_ap_index[p_tree_scope[index]][0] ;
    while ( count <= word_nums ) {
      for ( int j = p_access_port_start[index] ; j < p_access_port_start[index]+(ap_nums/num_threads) ; j++ ) {
        if ( parallel_memory[p_tree_scope[index]][parallel_ap_index[p_tree_scope[index]][j]] == num ) {
          a_t++ ;
          out = true ;
          break ;
        } // if
      } // for

      if ( out )
        break ;

      if ( count < word_nums ) {
        P_ShiftP(p_tree_scope[index]) ;
      } // if
      count++ ;
    } // while

    if ( !out ) {
      cout << "Out of range!!!" ; 
      cout << " index: " << index << ", and num: " << num << ":::" ;
    } // if 
    return shift_count ;
  } // Paraller_Read() */

  void Parallel_Read( double num, int index, int num_threads ) {
    bool out = false ;
    int count, tree_scope = p_tree_scope[index] + ( num / PR_placement_limit ), a_t = 0 ;
    shift_count = 0 ;

    count = parallel_ap_index[tree_scope][0] ;
    while ( count <= word_nums ) {
      for ( int j = p_access_port_start[index] ; j < p_access_port_start[index]+(ap_nums/num_threads) ; j++ ) {
        ap_access_time[tree_scope][j]++ ;
        if ( parallel_memory[tree_scope][parallel_ap_index[tree_scope][j]] == num ) {
          a_t++ ;
          out = true ;
          break ;
        } // if
      } // for

      if ( out )
        break ;

      if ( count < word_nums ) {
        P_ShiftP(tree_scope) ;
      } // if
      count++ ;
    } // while

    if ( !out ) {
      cout << "Out of range!!!" ; 
      cout << " index: " << index << ", and num: " << num << ":::" ;
    } // if 

    // cout << num << ": " << shift_count << endl ;
    p_track_shift[tree_scope]+=shift_count ;
    p_track_at[tree_scope]+=a_t ;
  } // Paraller_Read()

  int Level_Tree_Read( double num, int index, int s_c ) {
    bool out = false ;
    int count ;
    s_c = 0 ;

    for ( int i = 0 ; i < tree_scope[index].size() ; i++ ) {
      count = ap_index[tree_scope[index][i]][0] ;
      while ( count <= word_nums ) {
        for ( int j = 0 ; j < ap_nums ; j++ ) {
          if ( memory[tree_scope[index][i]][ap_index[tree_scope[index][i]][j]] == num ) {
            access_time++ ;
            out = true ;
            break ;
          } // if
        } // for

        if ( out )
          break ;

        if ( count < word_nums )
          Thread_ShiftP(tree_scope[index][i], s_c) ;
        count++ ;
      } // while

      if ( out )
        break ;

      if ( i == tree_scope[index].size()-1 && !out ) {
        cout << "Out of range!!!" ; 
        cout << " index: " << index << ", and num: " << num << ":::" << endl ;
      } // if 
    } // for 

    return s_c ;
  } // Level_Tree_Read()

  /* int Level_Tree_Read( double num, int index, int ap_start_index, int word_nums_index, int end_index, int ap_end_index, int word_nums_end_index ) { // FOR COMPACT ONLY
    bool out = false ;
    int count, inner_ap_index, inner_ap_limit ;
    int inner_word_nums_limit ;
    shift_count = 0 ;

    int i = index ; 
    while( i <= end_index ) {
      count = ap_index[i][0] ;
      inner_ap_index = last_apindex % ap_nums ;

      if ( i == end_index ) 
        inner_word_nums_limit = word_nums_end_index ;
      else 
        inner_word_nums_limit = word_nums ;

      while ( count <= inner_word_nums_limit ) {
        if ( i == end_index && count == word_nums_end_index )
          inner_ap_limit = ap_end_index + 1 ;
        else
          inner_ap_limit = ap_nums ;

        for ( int j = inner_ap_index ; j < inner_ap_limit ; j++ ) {
          // cout << "Track id: " << i << ", access port id: " << j << ", and value: " << memory[i][ap_index[i][j]] << endl ;
          if ( memory[i][ap_index[i][j]] == num ) {
            out = true ;
            last_apindex++ ;
            break ;
          } // if
          last_apindex++ ;
        } // for

        if ( out )
          break ;

        if ( count < word_nums ) 
          ShiftP(i) ;
        count++ ;
        inner_ap_index = 0 ;
      } // while

      if ( out )
        break ;

      if ( i == end_index && !out ) {
        cout << "***************Out of range!!!*****************" << endl ; 
        cout << " index: " << index << ", and num: " << num << ":::" << endl ;
      } // if 
      i++ ;
    } // while 

    return shift_count ;
  } // Level_Tree_Read() */

  void Reset() {
    memory.clear() ;
    total_track = 0 ;
    shift_count = 0 ;
    word_nums = 0 ;
    ap_nums = 0 ;
    ap_index.clear() ;
    tree_scope.clear() ;
  } // Reset()

  // ----------------------------------------above are shift code --------------------------------------------
  std::vector<std::vector<double>> buffer, tempbuffer, NLFbuffer ;

  std::vector<std::vector<std::vector<int>>> searchList, parallel_searchList ;

  int buffersize, PR_placement_limit ;

  void countSearchTime(std::vector<std::unique_ptr<Tree>> trees) {
    for ( int i = 0 ; i < searchList.size() ; i++ ) {
      for ( int j = 0 ; j < searchList[i].size() ; j++ ) {
        for ( int k = 0 ; k < searchList[i][j].size() ; k++ ) {
          trees[i]->node_list[searchList[i][j][k]].usage++ ;
        } // for
      } // for
    } // for
  } // countSearchTime()

  void printsearchList() {
    for ( int i = 0 ; i < searchList.size() ; i++ ) {
      for ( int j = 0 ; j < searchList[i].size() ; j++ ) {
        cout << "Tree" << i << "'s " ;
        if ( j == 0 )
          cout << j+1 << "st searching list:" << endl ;
        else if ( j == 1 )
          cout << j+1 << "nd searching list:" << endl ;
        else if ( j == 2 )
          cout << j+1 << "rd searching list:" << endl ;
        else 
          cout << j+1 << "th searching list:" << endl ;

        for ( int k = 0 ; k < searchList[i][j].size() ; k++ ) {
          cout << searchList[i][j][k] ;
          if ( k != searchList[i][j].size()-1 )
            cout << ", " ;
        } // for
        cout << endl ;
      } // for
      cout << endl ;
    } // for
  } // printsearchList()

  int  returnBufferSize() {
    int size = 0 ;
    for ( int i = 0 ; i < buffer.size() ; i++ ) {
      for ( int j = 0 ; j < buffer[i].size() ; j++ )
        size++ ;
    } // for

    return size ;
  } // returnBufferSize() 

  int  returnNLFBufferSize() {
    int size = 0 ;
    for ( int i = 0 ; i < NLFbuffer.size() ; i++ ) {
      for ( int j = 0 ; j < NLFbuffer[i].size() ; j++ )
        size++ ;
    } // for

    return size ;
  } // returnNLFBufferSize() 

  void print( int i ) {
    cout << "Buffer" << i+1 << "'s size: " << buffer[i].size() << endl ;
  } // print()

  void printBuffer() {
    for ( int i = 0 ; i < buffer.size() ; i++ ) {
      cout << "Buffer" << i+1 << ": " ;
      for ( int j = 0 ; j < buffer[i].size() ; j++ ) {
        cout << buffer[i][j] ;
        if ( j < buffer[i].size() - 1 )
          cout << ", " ;
      } // for
      
      // cout << endl << "Buffer" << i+1 << "'s size: " << buffer[i].size() ;
      cout << endl ; 
    } // for
  } // printBuffer()

  void printNLFBuffer() {
    for ( int i = 0 ; i < NLFbuffer.size() ; i++ ) {
      cout << "NLF-Buffer" << i+1 << ": " ;
      for ( int j = 0 ; j < NLFbuffer[i].size() ; j++ ) {
        cout << NLFbuffer[i][j] ;
        if ( j < NLFbuffer[i].size() - 1 )
          cout << ", " ;
      } // for
      
      // cout << endl << "Buffer" << i+1 << "'s size: " << buffer[i].size() ;
      cout << endl ; 
    } // for
  } // printBuffer()

  void printTBuffer() {
    for ( int i = 0 ; i < tempbuffer.size() ; i++ ) {
      cout << "TBuffer" << i+1 << ": " ;
      for ( int j = 0 ; j < tempbuffer[i].size() ; j++ ) {
        cout << tempbuffer[i][j] ;
        if ( j < tempbuffer[i].size() - 1 )
          cout << ", " ;
      } // for
      
      // cout << endl << "Buffer" << i+1 << "'s size: " << buffer[i].size() ;
      cout << endl ; 
    } // for
  } // printTBuffer()
} ;


} // namespace ranger

#endif /* FOREST_H_ */
