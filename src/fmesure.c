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
#include "fmesure.h"

double fmesure(size_t* out, const seq_t* seq, size_t Y ) {


	// Approx phrase par phrase de la f-mesure
	// fmesure = 2 * precision * rappel / (precision + rappel)

	const size_t T = seq->len;
	
	int* posCount = malloc(Y * sizeof(int));
	int* tagCount = malloc(Y * sizeof(int)); 
	int* labCount = malloc(Y * sizeof(int));

	for (size_t y = 0 ; y < Y ; y++ )
		posCount[y] = tagCount[y] = labCount[y] = 0 ;

	for(size_t t = 0 ; t < T ; t++ ) {
		size_t y = out[t]; 
		size_t yt = seq->pos[t].lbl; 
		labCount[yt]++; //Nombre de fois que chaque label apparait dans la ref
		tagCount[y]++; //Nombre de fois que chaque label est étiqueté
		if(y == yt)
			posCount[y]++; //Nombre de fois que l'étiquetage est bon
	};

	double precision = 0;
	double recall = 0;

	for(size_t y=0 ; y < Y ; y++) {
		//TODO : tag(y) ou lab(y) = 0
		double t = tagCount[y];
		double l = labCount[y];
		if (t)
			precision += (double) posCount[y] / t;
		if (l)	
			recall += (double) posCount[y] / l;

	}

	return (2 * precision * recall) / (Y * (precision + recall)) ;

}

double nfmesure(size_t N,size_t n, size_t out[][N], const seq_t* seq, size_t Y ) {

	// Approx phrase par phrase de la f-mesure
	// fmesure = 2 * precision * rappel / (precision + rappel)

	const size_t T = seq->len;
	
	int* posCount = malloc(Y * sizeof(int));
	int* tagCount = malloc(Y * sizeof(int)); 
	int* labCount = malloc(Y * sizeof(int));

	for (size_t y = 0 ; y < Y ; y++ )
		posCount[y] = tagCount[y] = labCount[y] = 0 ;

	for(size_t t = 0 ; t < T ; t++ ) {
		size_t y = out[t][n]; 
		size_t yt = seq->pos[t].lbl; 
		labCount[yt]++; //Nombre de fois que chaque label apparait dans la ref
		tagCount[y]++; //Nombre de fois que chaque label est étiqueté
		if(y == yt)
			posCount[y]++; //Nombre de fois que l'étiquetage est bon
	};

	double precision = 0;
	double recall = 0;

	for(size_t y=0 ; y < Y ; y++) {
		//TODO : tag(y) ou lab(y) = 0
		double t = tagCount[y];
		double l = labCount[y];
		if (t)
			precision += (double) posCount[y] / t;
		if (l)	
			recall += (double) posCount[y] / l;

	}

	return (2 * precision * recall) / (Y * (precision + recall)) ;

}
