import { Component, OnInit } from '@angular/core';
import PeopleJSON from '../../assets/people.json';  

interface Person {  
  name: String;
  position: String;
  image: String;
  url?: String;
}

@Component({
  selector: 'app-people',
  templateUrl: './people.component.html',
  styleUrls: ['./people.component.css']
})
export class PeopleComponent implements OnInit {

  people: Person[] = PeopleJSON;  

  constructor() { }

  ngOnInit(): void {
  }

}