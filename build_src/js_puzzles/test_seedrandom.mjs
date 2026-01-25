
import { SumOfDigits } from './generators/index.js';
const gen = new SumOfDigits(42);
gen.genRandom();
console.log(JSON.stringify(gen.instances));

