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

  // ----------------------------------------below is my own code --------------------------------------------
  // ----------------------------------------below is shift code ---------------------------------------------

  vector<vector<double>> memory, NLFmemory ;
  vector<vector<int>> ap_index ;
  vector<int> tree_scope, tree_scope_end, access_port_start ;
  vector<int> access_port_end, wordnum_start, wordnum_end ;
  int total_track = 0, shift_count = 0, word_nums = 0 ;
  int ap_nums = 0, last_index = 0, remainder_count = 0 ;
  int last_apindex ;
  bool NLF_Mode ;

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

  void PrintSfift() {
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
  } // Print()ap_index

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

  
  void Level_Tree_Write( vector<double> input, int index ) {
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
  } // Level_Tree_Write 
  
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

  int Level_Tree_Read( double num, int index, int ap_start_index, int word_nums_index, int end_index, int ap_end_index, int word_nums_end_index ) {
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
  } // Level_Tree_Read() 

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

  std::vector<std::vector<std::vector<int>>> searchList ;

  int buffersize ;

  void countSearchTime() {
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

  // ----------------------------------------above is my own code --------------------------------------------
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

} // namespace ranger

#endif /* FOREST_H_ */
