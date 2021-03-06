/*
 *      Wapiti - A linear-chain CRF tool
 *
 * Copyright (c) 2009-2011  CNRS
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "options.h"
#include "progress.h"
#include "sequence.h"
#include "tools.h"
#include "vmath.h"

#include "decoder.h"
#include "trainers.h"
#include "fmesure.h"

//Algorithme MIRA**/

void trn_mira_single(mdl_t *mdl) {
	const size_t  Y = mdl->nlbl;
	const size_t  F = mdl->nftr;
	const int     S = mdl->train->nseq;
	const int     K = mdl->opt->maxiter;
	const double C = mdl->opt->mira.C;
	double 	*w = mdl->theta;	
	
	//wSum : somme de tous les poids
	double* wSum = xmalloc(F*sizeof(double));
	// Nombre de fois que l'élément n'a pas été mis à jour
	size_t* wCache = xmalloc(F*sizeof(size_t));
	for(size_t i = 0 ; i < F ; i++) {
		wSum[i] = w[i];
		wCache[i] = 0;
	}
	//Nombre total de mises à jour de w
	size_t N = 1;


	// We will process sequences in random order in each iteration, so we
	// will have to permute them. The current permutation is stored in a
	// vector called <perm> shuffled at the start of each iteration. We
	// just initialize it with the identity permutation.
	int *perm = xmalloc(sizeof(int) * S);
	for (int s = 0; s < S; s++)
		perm[s] = s;

	for (int k = 0 ; k < K && !uit_stop; k++) { 
		//pour un nombre maxiter de fois

/*		// First we shuffle the sequence by making a lot of random swap
		// of entry in the permutation index.
		for (int s = 0; s < S; s++) {
			const int a = rand() % S;
			const int b = rand() % S;
			const int t = perm[a];
			perm[a] = perm[b];
			perm[b] = t;
		}
*/
		// And so, we can process sequence in a random order
		for (int sp = 0; sp < S && !uit_stop; sp++) {
printf("\n sp = %d ",sp);	
			const int s = perm[sp];
			const seq_t *seq = mdl->train->seq[s];
			int T = seq->len;
			size_t* out = xmalloc(T * sizeof(size_t)); //tableau 
			//de labels
			tag_viterbi(mdl, seq, out,NULL,NULL);
			bool differents = false;
			//On commence par regarder si le meilleur (out) est 
			//la référence (seq)
			size_t featCount = 0;
			double featSum = 0;
			for(int t = 0 ; t < T ; t++) {
				//Pour chaque unité dans les séquences : 
				//(les deux en ont autant)
				if (out[t] != (seq->pos[t]).lbl ) { 
					//si les deux labels sont différents
					differents = true;
				// Norme de \delta h : pour chaque, si égal 0 si différents +1
				// = count
					featCount++;	
				}
			}
			if (differents) { 
				// si  y != y(s) (le meilleur n'est pas 
				// la référence)
				// theta = theta + marge = theta + 
				// alpha*(features(y(t),x(t))- features(y,x(s)))
				//Pour chaque position t dans y, pour 
				//chaque feature activée par (y,s,x) :
				// On prend en compte les features 
				// unigrammes du premier mot

				//Alpha dépend de ....
				//On a alpha = max(0,min(C,(loss(out,seq) 
				// - \delta H_(t-1))/norme(\delta h_(t-1))

				//On calcule alpha
				//L = 1 - fmesure(out, pos, T)
				// \delta H : pour chaque si égal 0 si différents +(différence
				// des poids)

				// Pour le premier mot on ne regarde que les feat. unigrammes
				const pos_t* pos = &(seq->pos[0]);
				size_t y = out[0]; 
				size_t ys = pos->lbl;
				for(size_t p = 0 ; p < pos->ucnt ; p++) {
					const size_t o = pos->uobs[p];
					featSum += w[mdl->uoff[o] + ys] - w[mdl->uoff[o] + y]; 
				}
				//Pour tous les mots suivants, on regarde 
				//à la fois les unigrammes et les bigrammes
				for(int t = 1 ; t < T ; t++) { 
					pos = &(seq->pos[t]);
					y = out[t]; 
					ys = pos->lbl;
					size_t yp = out[t-1]; 
					size_t yps = seq->pos[t-1].lbl; 
					for(size_t p = 0 ; p < pos->ucnt ; p++) {
						const size_t o = pos->uobs[p];
					featSum += w[mdl->uoff[o] + ys] - w[mdl->uoff[o] + y]; 
					}
					for(size_t p = 0 ; p < pos->bcnt  && !uit_stop ; p++) {
						const size_t o = pos->bobs[p];
						size_t d = Y*yp + y;
						size_t ds = Y*yps + ys;
					featSum += w[mdl->boff[o] + ds] - w[mdl->boff[o] + d];
					} 
				}
				size_t (*out_2d)[T][1] = (void*) out;
				double delta = (1 - nfmesure(1,0,*out_2d,seq,Y) - featSum) / (double) featCount;
printf(" featsum : %f ", featSum);
printf(" featcount : %lu ", featCount);

			//	double delta = (1 - fmesure(out,seq,Y) - featSum) / (double) featCount;
			printf("delta : %g ", delta);
				double alpha = (delta < C) ? ((delta > 0) ? delta : 0) : C ;
			// Maintenant qu'on a calculé alpha, on peut appliquer l'update perceptron
				pos = &(seq->pos[0]);
				y = out[0]; 
				ys = pos->lbl;
				for(size_t p = 0 ; p < pos->ucnt ; p++) {
					const size_t o = pos->uobs[p];
					const size_t j = mdl->uoff[o] + y;		
					const size_t js = mdl->uoff[o] + ys;		
					wSum[j] += (N - wCache[j]) * w[j];
					wSum[js] += (N - wCache[js]) * w[js];
					w[js] += alpha;
					w[j] -= alpha;
					wCache[js] = N;
					wCache[j] = N;
				}
				//Pour tous les mots suivants, on regarde 
				//à la fois les unigrammes et les bigrammes
				for(int t = 1 ; t < T ; t++) { 
					const pos_t *pos = &(seq->pos[t]);
					const size_t y = out[t]; 
					const size_t ys = pos->lbl;
					const size_t yp = out[t-1]; 
					const size_t yps = seq->pos[t-1].lbl; 
					for(size_t p = 0 ; p < pos->ucnt ; p++) {
						const size_t o = pos->uobs[p];
						const size_t j = mdl->uoff[o] + y;		
						const size_t js = mdl->uoff[o] + ys;		
						wSum[j] += (N - wCache[j]) * w[j];
						wSum[js] += (N - wCache[js]) * w[js];
						w[js] += alpha;
						w[j] -= alpha;
						wCache[js] = N;
						wCache[j] = N;
					}
					for(size_t p = 0 ; p < pos->bcnt ; p++) {
						const size_t o = pos->bobs[p];
						const size_t j = mdl->boff[o] + Y*yp + y;		
						const size_t js = mdl->boff[o] + Y*yps + ys;		
						wSum[j] += (N - wCache[j]) * w[j];
						wSum[js] += (N - wCache[js]) * w[js];
						w[js] += alpha;
						w[j] -= alpha;
						wCache[js] = N;
						wCache[j] = N;
					} 
				}
				N++;
			}
			free(out);
		}		
		for(size_t i = 0 ; i < F ; i++)
			w[i] =  (wSum[i] + (N - wCache[i]) * w[i]) / (double) N;
		// Repport progress back to the user
		if (!uit_progress(mdl, k + 1, -1.0))
			break;
	}
	free(wSum);
	free(wCache);
	free(perm);
};
/*
theta : contient toutes les features présentes dans le train (tous les tests sur les observations, * Y feat unigrammes * YY feat bigrammes)
qui??? génère les feat pour une phrase (d'entrainement...) donnée ?
-> défini dans /sequence.h/, |uobs| est créé dans _dotrain_

l'algo d'entraînement est appelé dans dotrain(mdl_t *mdl), avec comme argument un modèle (où les données d'entraînement ont été chargées : on a les uobs , les bobs dans seq->pos[i] : données de la séquence d'entraînement à la position $i$, cf sequence.h


// Pour chaque position dans la séquence
for (int t = 0; t < T; t++) {
// On récupère les données à cette position
const pos_t *pos = &(seq->pos[t]);
// Pour chaque label possible
for (size_t y = 0; y < Y; y++) {
double sum = 0.0;
for (size_t n = 0; n < pos->ucnt; n++) {
const size_t o = pos->uobs[n];
sum += x[mdl->uoff[o] + y];
}
for (size_t yp = 0; yp < Y; yp++)
(*psi)[t][yp][y] = sum;
}
}

*/
