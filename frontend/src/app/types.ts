export type Candidate = {
  name: string;
  confidence: number;
};

export type PredictResult = {
  name: string;
  confidence: number;
  calories: number | string;
  portion: string;
  full_name: string;
  top3: Candidate[];
};
