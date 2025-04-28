#ifndef __ENV_SPOT_2D_H_
#define __ENV_SPOT_2D_H_

#include <memory> // std::unique_ptr
#include <iostream>
#include <unordered_map>
#include <map>
#include <utility>                   // For std::pair
#include <boost/heap/d_ary_heap.hpp> // heap
#include <Eigen/Dense>
#include <set>
#include "solar_planner/grid_map.h"
#include "solar_planner/planning_interface.h"

#include <spot/parseaut/public.hh>
#include <spot/twa/bdddict.hh>
#include <spot/twa/bddprint.hh>

namespace erl
{

  /**
   * @brief The PlanningSpot2D class implements PlanningInterface for a Linear Temporal Logic Environment and 2 Dimensions using spot.
   */
  class PlanningSpot2D : public PlanningInterface<std::array<int, 3>>
  {
    const Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic> &lmap_;
    const Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic> &occ_map_;
    std::unordered_map<uint16_t, std::vector<std::array<int, 2>>> label_to_xy_cell;

  public:
    Eigen::MatrixXd label_distance_g_; // num_labels x spec->num_sta()
                                       // const double *label_distance_g;

    std::unique_ptr<erl::GridMap<uint16_t>> MAP_ptr;
    spot::twa_graph_ptr automaton; // Logic specification
    std::map<std::pair<int, int>, std::set<int>> dictionary_of_labels;
    std::set<uint16_t> unique_labels;

    /**
     * @brief Constructs the PlanningSpot2D Environment.
     * @param lmap The label map.
     * @param occ_map The occupancy map.
     * @param MAP_ptr_in The input Map pointer.
     * @param spot_ptr The spot pointer.
     */
    PlanningSpot2D(const Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic> &lmap,
                   const Eigen::Matrix<uint16_t, Eigen::Dynamic, Eigen::Dynamic> &occ_map,
                   std::unique_ptr<erl::GridMap<uint16_t>> MAP_ptr_in,
                   spot::twa_graph_ptr spot_ptr)
        : lmap_(lmap), occ_map_(occ_map), MAP_ptr(std::move(MAP_ptr_in)), automaton(std::move(spot_ptr))
    {
      // initialize label_to_xy_cell
      for (int x = 0; x < MAP_ptr->size()[0]; ++x)
        for (int y = 0; y < MAP_ptr->size()[1]; ++y)
        {
          uint16_t label = lmap_(x, y);
          unique_labels.insert(label);

          auto it = label_to_xy_cell.find(label);

          // if element does not exist, insert it!
          if (it == label_to_xy_cell.end())
          {
            auto it1 = label_to_xy_cell.insert(std::make_pair(label,
                                                              std::vector<std::array<int, 2>>(1, {x, y})));
            it1.first->second.reserve(MAP_ptr->size()[0] * MAP_ptr->size()[1]);
          }
          else
            it->second.push_back({x, y});
        }

      // for every edge in the automaton
      for (auto &edge : automaton->edges())
      {
        // get the source and destination states
        int src = edge.src;
        int dst = edge.dst;


        // for every unique label
        for (uint16_t label_int : unique_labels)
        {
          uint16_t temp = label_int;
          // get the condition of the edge
          bdd check_formula = edge.cond;

          std::vector<int> label(automaton->ap().size(), 0); // Initialize vector with all 0s

          for (int i = automaton->ap().size() - 1; i >= 0 && temp > 0; --i)
          {
            label[i] = temp % 2; // Get the least significant bit
            temp /= 2;           // Shift the number right by dividing by 2
          }

          // Iterate through the APs of the label
          for (size_t i = 0; i < label.size(); ++i)
          {
            // Check if AP is true
            if (label[i])
            {
              // Substitutes true for the AP p_i in the formula
              check_formula = bdd_restrict(check_formula, bdd_ithvar(i)); // bdd_ithvar(i) creates a true bdd at location i, eg. <i:1>
            }
            else
            {
              // Substiutes false for the AP p_i in the formula
              check_formula = bdd_restrict(check_formula, !bdd_ithvar(i));
            }

            // if check formula is false then skip
            if (check_formula == bdd_false() || check_formula == bdd_true())
            {
              break;
            }
          }
          // Checks if the overall formula evaluates to true after substitution
          if (check_formula == bdd_true())
          {
            // insert the edge into the dictionary
            std::pair<int, int> key = {src, dst};
            //

            dictionary_of_labels[key].insert(label_int);
          }
        }
      }


      //  initialize label_distance_g_
      label_distance_g_ = std::move(label_cost2go());
    }

    /**
     * @brief get the successors of the current state from spot.
     * @param curr The current state to compute successors of.
     * @param succ The vector of successor states.
     * @param succ_cost The vector of costs of each successor.
     * @param action_idx The vector of indices of the actions leading to the successor.
     */
    void getSuccessors(const std::array<int, 3> &curr,        // x, y , q (automaton state)
                       std::vector<std::array<int, 3>> &succ, // x, y, q (automaton state)
                       std::vector<double> &succ_cost,        // cost of each successor
                       std::vector<int> &action_idx) const    // index of the action leading to the successor
    {
      succ.reserve(8);
      succ_cost.reserve(8);
      action_idx.reserve(8);

      for (int xShift = -1; xShift <= 1; ++xShift)
      {
        int xNeighbor = curr[0] + xShift;
        if (xNeighbor < 0 || xNeighbor >= MAP_ptr->size()[0])
          continue; // Skip out-of-bounds

        for (int yShift = -1; yShift <= 1; ++yShift)
        {
          if (xShift == 0 && yShift == 0)
            continue; // Skip the current node

          int yNeighbor = curr[1] + yShift;
          if (yNeighbor < 0 || yNeighbor >= MAP_ptr->size()[1])
            continue; // Skip out-of-bounds

          // filter by occupancy
          if (occ_map_(xNeighbor, yNeighbor) > 0)
            continue; // Skip occupied cells

          // neighbor label
          uint16_t label_int = lmap_(xNeighbor, yNeighbor);

          // get the next q
          int qNeighbor = next_q(curr[2], label_int);

          if (!is_sink_state(qNeighbor))
          {

            // add to successors
            succ.push_back({xNeighbor, yNeighbor, qNeighbor});

            double dx = xShift * MAP_ptr->res()[0];
            double dy = yShift * MAP_ptr->res()[1];
            double costMult = std::sqrt(dx * dx + dy * dy);
            succ_cost.push_back(costMult);

            action_idx.push_back((xShift + 1) + (yShift + 1) * 3); // trenary to decimal
          }
        }
      }
    }

    /**
     * @brief Indicates if a state s is the goal state.
     * @param s The state to check for goal conditions.
     * @return True if the state is at the goal.
     */
    inline bool isGoal(const std::array<int, 3> &state_coord) const
    {
      return automaton->state_is_accepting(state_coord[2]);
    }

    /**
     * @brief Computes a heuristic estimate from the state to the goal.
     * @param s The query state.
     * @return The heuristic value.
     */

    double getHeuristic(const std::array<int, 3> &state_coord) const
    {
      double h = std::numeric_limits<double>::infinity();

      // if goal state then cost is 0
      if (automaton->state_is_accepting(state_coord[2]))
      {
        return 0;
      }

      // if sink state then cost is inf
      if (is_sink_state(state_coord[2]))
      {
        return std::numeric_limits<double>::infinity();
      }

      for (unsigned int nq = 0; nq < automaton->num_states(); ++nq)
      {
        // skip self-loops
        if (state_coord[2] == (int)nq)
          continue;

        // labels that take us to next_q
        std::set<int> labs;

        // check if the dictionary has the key
        if (dictionary_of_labels.find(std::make_pair(state_coord[2], nq)) != dictionary_of_labels.end())
        {
          labs = dictionary_of_labels.at(std::make_pair(state_coord[2], nq));
        }

        // for every element in the labs set
        for (int lab_it : labs)
        {

          // distnace from point to set
          double c = std::numeric_limits<double>::infinity();
          auto it = label_to_xy_cell.find(lab_it);

          if (it == label_to_xy_cell.end())
            continue;

          size_t num_sta = it->second.size();
          for (size_t sta = 0; sta < num_sta; ++sta)
          {
            double dx = static_cast<double>(it->second[sta][0] - state_coord[0]) * MAP_ptr->res()[0];
            double dy = static_cast<double>(it->second[sta][1] - state_coord[1]) * MAP_ptr->res()[1];
            double dist2 = dx * dx + dy * dy;
            if (dist2 < c)
              c = dist2;
          }
          c = std::sqrt(c);

          double tentative_h = c + label_distance_g_(lab_it, nq);

          if (tentative_h < h)
            h = tentative_h;
        }
      }

      return h;
    }

    /**
     * @brief Converts the state to a linear index.
     * @param s The state to be converted.
     * @return The linear index.
     */
    inline size_t stateToIndex(const std::array<int, 3> &state_coord) const
    {
      return state_coord[0] + state_coord[1] * MAP_ptr->size()[0] + state_coord[2] * MAP_ptr->size()[0] * MAP_ptr->size()[1];
    }

  private:
    double label_distance(int l1, int l2) const
    {
      if (l1 == l2)
        return 0; // Distance from label to itself is always 0

      auto xy_arr1 = label_to_xy_cell.find(l1);
      if (xy_arr1 == label_to_xy_cell.end())
        return std::numeric_limits<double>::infinity(); // Distance to empty label is inf

      auto xy_arr2 = label_to_xy_cell.find(l2);
      if (xy_arr2 == label_to_xy_cell.end())
        return std::numeric_limits<double>::infinity(); // Distance to empty label is inf

      double min_d = std::numeric_limits<double>::infinity();

      size_t num_xy1 = xy_arr1->second.size();
      size_t num_xy2 = xy_arr2->second.size();

      for (size_t st1 = 0; st1 < num_xy1; ++st1)
        for (size_t st2 = 0; st2 < num_xy2; ++st2)
        {
          double dx = (xy_arr1->second[st1][0] - xy_arr2->second[st2][0]) * MAP_ptr->res()[0];
          double dy = (xy_arr1->second[st1][1] - xy_arr2->second[st2][1]) * MAP_ptr->res()[1];
          double d = dx * dx + dy * dy;
          if (d < min_d)
            min_d = d;
        }
      return std::sqrt(min_d);
    }

    typedef std::tuple<double, int, int> CostNodePair; // pair of a double cost and node number

    struct compare_cnpair
    {
      bool operator()(const CostNodePair &t1, const CostNodePair &t2) const
      {
        return std::get<0>(t1) > std::get<0>(t2);
      }
    };

    Eigen::MatrixXd label_cost2go() const
    {
      using PriorityQueue = boost::heap::d_ary_heap<CostNodePair, boost::heap::mutable_<true>,
                                                    boost::heap::arity<2>, boost::heap::compare<compare_cnpair>>;

      typedef PriorityQueue::handle_type heapkey;

      unsigned num_lab = std::pow(2, automaton->ap().size());

      // transition cost between labels
      Eigen::MatrixXd costL;
      costL.setConstant(num_lab, num_lab, std::numeric_limits<double>::infinity());

      // gvalues (accumulated cost)
      Eigen::MatrixXd g;
      g.setConstant(num_lab, automaton->num_states(), std::numeric_limits<double>::infinity());

      // closed list
      Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> cl;
      cl.setZero(num_lab, automaton->num_states());

      // store heapkeys
      Eigen::Matrix<heapkey, Eigen::Dynamic, Eigen::Dynamic> hk(num_lab, automaton->num_states());

      // open list
      PriorityQueue pq;
      Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> op;
      op.setZero(num_lab, automaton->num_states());

      // Initialize the g-values of all accepting states to 0 and add them to OPEN
      unsigned n = automaton->num_states();
      for (unsigned s = 0; s < n; ++s)
      {
        if (automaton->state_is_accepting(s))
        {
          g.col(s) = Eigen::VectorXd::Zero(num_lab);
          for (unsigned lab = 0; lab < num_lab; ++lab)
          {
            hk(lab, s) = pq.push(std::make_tuple(0.0, lab, s));
            op(lab, s) = true;
          }
        }
      }

      while (!pq.empty())
      {
        // get node with min gval
        CostNodePair currNode = pq.top();
        pq.pop();
        int lCurr = std::get<1>(currNode);
        int qCurr = std::get<2>(currNode);
        op(lCurr, qCurr) = false;
        cl(lCurr, qCurr) = true;

        // find predecessors
        for (int qPred = 0; qPred < automaton->num_states(); ++qPred)
        {
          if (qPred == qCurr)
            continue; // skip self-loops

          if (next_q(qPred, lCurr) != qCurr && next_q(qPred, lCurr) != -1)
            continue;

          for (unsigned int lPred = 0; lPred < num_lab; ++lPred)
          {
            if (cl(lPred, qPred))
              continue; // skip closed

            // compute transition cost if needed
            if (std::isinf(costL(lPred, lCurr)))
            {
              costL(lPred, lCurr) = label_distance(lPred, lCurr);
              costL(lCurr, lPred) = costL(lPred, lCurr);
            }

            double tentative_g = costL(lPred, lCurr) + g(lCurr, qCurr);

            if (tentative_g < g(lPred, qPred))
            {
              g(lPred, qPred) = tentative_g;
              if (op(lPred, qPred))
                pq.increase(hk(lPred, qPred), std::make_tuple(tentative_g, lPred, qPred));
              else
                hk(lPred, qPred) = pq.push(std::make_tuple(tentative_g, lPred, qPred));
            }
          }
        }
      }
      return g;
    }

    bool is_sink_state(int state) const
    {
      if (automaton->state_is_accepting(state))
        return false;

      for (const auto &nt : automaton->out(state))
      {
        if (nt.dst != state)
        {
          return false;
        }
      }
      return true;
    }

    int next_q(int q, unsigned int label_int) const
    {

      // for every state in the automaton
      unsigned n = automaton->num_states();
      for (unsigned s = 0; s < n; ++s)
      {
        if (dictionary_of_labels.find(std::make_pair(q, s)) != dictionary_of_labels.end())
        {
          if (dictionary_of_labels.at(std::make_pair(q, s)).find(label_int) != dictionary_of_labels.at(std::make_pair(q, s)).end())
          {
            return s;
          }
        }

      }
      return -1; // TODO: THIS MAY NOT BE VALID
    }
  };
}

#endif // __ENV_SPOT_2D_H_
