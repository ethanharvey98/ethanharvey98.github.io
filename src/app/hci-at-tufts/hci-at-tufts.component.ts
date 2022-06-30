import { Component, OnInit } from '@angular/core';
import CoursesJSON from '../../assets/courses.json';  

interface Course {  
  department: String;  
  code: String;
  title: String;
}

@Component({
  selector: 'app-hci-at-tufts',
  templateUrl: './hci-at-tufts.component.html',
  styleUrls: ['./hci-at-tufts.component.css']
})
export class HciAtTuftsComponent implements OnInit {

  courses: Course[] = CoursesJSON;  

  constructor() { }

  ngOnInit(): void {
  }

}
