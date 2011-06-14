size_t fmesure(size_t* out, seq_t* seq, size_t Y ) {

	// Approx phrase par phrase de la f-mesure
	// fmesure = 2 * precision * rappel / (precision + rappel)

	const size_t T = seq->len;
	
	int* posCount = malloc(Y * sizeof(int));
	int* tagCount = malloc(Y * sizeof(int)); 
	int* labCount = malloc(Y * sizeof(int));

	for (int y = 0 ; y < Y ; y++ )
		posCount[y] = tagCount[y] = labCount[y] = 0 ;

	for(int t = 0 ; t < T ; t++ ) {
		size_t y = out[t]; 
		size_t yt = seq->pos[t].lbl; 
		labCount[yt]++; //Nombre de fois que chaque label apparait dans la ref
		tagCount[y]++; //Nombre de fois que chaque label est étiqueté
		if(y == yt)
			posCount[y]++; //Nombre de fois que l'étiquetage est bon
	};

	double precision = double recall = 0;

	for(int y=0 ; y < Y ; y++) {
		//TODO : tag(y) ou lab(y) = 0
		precision += pos[y] / tag[y];
		recall += pos[y] / lab[y];

	}

	return (2 * precision * recall) / (Y * (precision + recall)) ;

}
