/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>

#include "particle_filter.h"
#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 1000;
    
    
    std::default_random_engine gen;
    std::normal_distribution<double> x_dist(x,std[0]);
    std::normal_distribution<double> y_dist(y,std[1]);
    std::normal_distribution<double> theta_dist(theta,std[2]);

    for(int i = 0; i < num_particles; i++){

        Particle particle = {
            	i,//int id;
                x_dist(gen),//double x;
                y_dist(gen),//double y;
                theta_dist(gen),//double theta;
                1//double weight; 
        };
        particles.push_back(particle);        
    }
    
    is_initialized = true;
    std::cout << "Particle filter initialized with " << particles.size() <<" Particles." <<std::endl;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    std::default_random_engine gen;

    

    for(auto& p: particles){
        
        std::normal_distribution<double> x_dist(p.x,std_pos[0]);
        std::normal_distribution<double> y_dist(p.y,std_pos[1]);
        std::normal_distribution<double> theta_dist(p.theta,std_pos[2]);
        double x = x_dist(gen);
        double y = y_dist(gen);
        double theta = theta_dist(gen);

        
        if(yaw_rate != 0){
            p.x = x +  velocity/yaw_rate * 
                            (sin(theta+ delta_t * yaw_rate) - sin(theta));
            p.y = y+ velocity/yaw_rate * 
                            (cos(theta) - cos(theta + delta_t * yaw_rate)); 
            p.theta =  theta + delta_t * yaw_rate;
        }else{
            p.x +=  velocity * delta_t * cos(p.theta);
            p.y +=  velocity * delta_t * sin(p.theta);
        }


    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for(auto& pl : predicted){
        std::vector<double> distances;
        for(auto& ol : observations){
            distances.push_back(sqrt(pow((pl.x-ol.x),2.0) + pow((pl.y-ol.y),2.0)));
        }
        std::vector<double>::iterator result = std::min_element(std::begin(distances),std::end(distances));
        //pl.id = observations[std::distance(std::begin(distances), result)];
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

    
    // for every particle
    for(auto& p : particles){
        
        // find LM in particles sensor_range ( wold_frame ) 
        std::vector<LandmarkObs> lms_pred;
        for(auto const& lm: map_landmarks.landmark_list){
            const double distance = dist(lm.x_f, lm.y_f, p.x, p.y);
            if(distance < sensor_range){
                LandmarkObs pred_lm;
                pred_lm.id = lm.id_i;
                pred_lm.x = lm.x_f;
                pred_lm.y = lm.y_f;
                lms_pred.push_back(pred_lm);    
            }
        }
        

        // transform observations into world frame
        std::vector<LandmarkObs> lms_obsv;
        for(LandmarkObs obs: observations){
            LandmarkObs newObs;
            newObs.x = obs.x * cos(p.theta) - obs.y * sin(p.theta) + p.x;
            newObs.y = obs.x * sin(p.theta) + obs.y * cos(p.theta) + p.y;
            lms_obsv.push_back(newObs);
        }

        std::vector<LandmarkObs> predicted;
        std::vector<LandmarkObs> observed;
        // assoziate lms_obsv with lms_pred
        
        for(auto const& lm_obs:lms_obsv){
            double distance_min = INFINITY; // init comparison
            LandmarkObs nearestLm;
            for(auto const& lm_pred: lms_pred){
                const double distance = dist(lm_pred.x,lm_pred.y,lm_obs.x,lm_obs.y);
                if (distance < distance_min){
                    distance_min = distance;
                    nearestLm = lm_pred;
                }
            }
            observed.push_back(lm_obs);
            predicted.push_back(nearestLm);

        }
        
        

        // calculate weight 
        double w = 1.0;
        for(int i = 0; i < observed.size(); i++){
            LandmarkObs l1 = predicted[i];
            LandmarkObs l2 = observed[i];
            double p =(1 / (2 * M_PI *std_landmark[0]*std_landmark[1]))
                       * std::exp(-0.5 * (
                       ((l2.x-l1.x)*(l2.x-l1.x)/(std_landmark[0]*std_landmark[0]))
                       + ((l2.y-l1.y)*(l2.y-l1.y)/(std_landmark[1]*std_landmark[1]))));
            w *= p;
        }
        p.weight = w;
    }
    

}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    std::vector<double> weights;
    double sum = 0;

    for(auto const& p:particles){
        weights.push_back(p.weight);
        sum += p.weight;
    }
    // normalize
    for(auto& w:weights ){
        w /= sum;
    }
    std::default_random_engine gen;
    std::discrete_distribution<> distribution(weights.begin(), weights.end());
    std::vector<Particle> particles_sampled;
    for(int i = 0; i < num_particles; i++){
        int weighted_index = distribution(gen);
        particles_sampled.push_back(particles.at(weighted_index));
    }
    particles = std::move(particles_sampled);
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
